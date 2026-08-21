package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	listenAddr  = "127.0.0.1:11435"
	nvidiaURL   = "https://integrate.api.nvidia.com/v1/chat/completions"
	nvidiaModel = "meta/muse-glimmer-30b"
)

type ResponsesRequest struct {
	Model             string          `json:"model"`
	Input             json.RawMessage `json:"input"`
	Instructions      json.RawMessage `json:"instructions,omitempty"`
	MaxOutputTokens   int             `json:"max_output_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	Tools             json.RawMessage `json:"tools,omitempty"`
	ToolChoice        json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
}

type ChatRequest struct {
	Model             string        `json:"model"`
	Messages          []ChatMessage `json:"messages"`
	Temperature       *float64      `json:"temperature,omitempty"`
	TopP              *float64      `json:"top_p,omitempty"`
	MaxTokens         int           `json:"max_tokens,omitempty"`
	Stream            bool          `json:"stream,omitempty"`
	Tools             []ChatTool    `json:"tools,omitempty"`
	ToolChoice        any           `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool         `json:"parallel_tool_calls,omitempty"`
}

type ChatTool struct {
	Type     string       `json:"type"`
	Function ChatFunction `json:"function"`
}

type ChatFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters any    `json:"parameters,omitempty"`
}

type ChatMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	ToolCalls  []ChatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type ChatToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ChatCallFunc `json:"function"`
}

type ChatCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ResponsesToolCall struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status"`
}

func main() {
	if os.Getenv("NVIDIA_API_KEY") == "" {
		log.Fatal("NVIDIA_API_KEY is not set")
	}

	http.HandleFunc("/v1/responses", handleResponses)
	http.HandleFunc("/v1/models", handleModels)

	server := &http.Server{
		Addr:              listenAddr,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("NVIDIA Codex proxy")
	log.Printf("Listening on http://%s", listenAddr)
	log.Printf("NVIDIA endpoint: %s", nvidiaURL)
	log.Printf("Model: %s", nvidiaModel)

	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	response := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"id":       nvidiaModel,
				"object":   "model",
				"owned_by": "nvidia",
			},
		},
	}

	_ = json.NewEncoder(w).Encode(response)
}

func writeSSEEvent(w http.ResponseWriter, event string, data any) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Printf("SSE marshal error: %v", err)
		return
	}

	fmt.Fprintf(w, "event: %s\n", event)
	fmt.Fprintf(w, "data: %s\n\n", jsonData)
}

func handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	log.Printf("Incoming Responses request: %d bytes", len(body))

	var req ResponsesRequest

	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(
			w,
			"invalid Responses JSON: "+err.Error(),
			http.StatusBadRequest,
		)
		return
	}

	messages, err := convertInput(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	tools, err := convertTools(req.Tools)
	if err != nil {
		http.Error(
			w,
			"invalid tools: "+err.Error(),
			http.StatusBadRequest,
		)
		return
	}

	var toolChoice any

	if len(req.ToolChoice) > 0 {
		if err := json.Unmarshal(req.ToolChoice, &toolChoice); err != nil {
			http.Error(
				w,
				"invalid tool_choice: "+err.Error(),
				http.StatusBadRequest,
			)
			return
		}

		toolChoice = convertToolChoice(toolChoice)
	}

	chatReq := ChatRequest{
		Model:             nvidiaModel,
		Messages:          messages,
		Temperature:       req.Temperature,
		TopP:              req.TopP,
		Stream:            false,
		Tools:             tools,
		ToolChoice:        toolChoice,
		ParallelToolCalls: req.ParallelToolCalls,
	}

	if req.MaxOutputTokens > 0 {
		chatReq.MaxTokens = req.MaxOutputTokens
	}

	chatBody, err := json.Marshal(chatReq)
	if err != nil {
		http.Error(
			w,
			"failed to encode NVIDIA request: "+err.Error(),
			http.StatusInternalServerError,
		)
		return
	}

	log.Printf(
		"Sending to NVIDIA: model=%s messages=%d tools=%d body=%d bytes",
		nvidiaModel,
		len(messages),
		len(tools),
		len(chatBody),
	)

	nvidiaReq, err := http.NewRequest(
		http.MethodPost,
		nvidiaURL,
		bytes.NewReader(chatBody),
	)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	nvidiaReq.Header.Set(
		"Authorization",
		"Bearer "+os.Getenv("NVIDIA_API_KEY"),
	)

	nvidiaReq.Header.Set("Content-Type", "application/json")
	nvidiaReq.Header.Set("Accept", "application/json")

	resp, err := http.DefaultClient.Do(nvidiaReq)
	if err != nil {
		http.Error(
			w,
			"NVIDIA request failed: "+err.Error(),
			http.StatusBadGateway,
		)
		return
	}

	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(
			w,
			"failed to read NVIDIA response: "+err.Error(),
			http.StatusBadGateway,
		)
		return
	}

	log.Printf(
		"NVIDIA response: HTTP %d, %d bytes",
		resp.StatusCode,
		len(responseBody),
	)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(responseBody)
		return
	}

	var completion map[string]any

	if err := json.Unmarshal(responseBody, &completion); err != nil {
		http.Error(
			w,
			"invalid NVIDIA response: "+err.Error(),
			http.StatusBadGateway,
		)
		return
	}

	toolCalls := extractToolCalls(completion)
	outputText := extractAssistantText(completion)

	log.Printf(
		"NVIDIA result: text=%d chars tool_calls=%d",
		len(outputText),
		len(toolCalls),
	)

	if len(toolCalls) > 0 {
		for _, call := range toolCalls {
			log.Printf(
				"Tool call: id=%s name=%s arguments=%s",
				call.ID,
				call.Function.Name,
				call.Function.Arguments,
			)
		}
	}

	if req.Stream {
		writeResponsesSSE(w, outputText, toolCalls)
		return
	}

	response := convertChatResponse(completion)

	w.Header().Set("Content-Type", "application/json")

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("response encoding error: %v", err)
	}
}

/*
	Convertit les outils Responses API de Codex vers
	le format tools de Chat Completions.
*/
func convertTools(raw json.RawMessage) ([]ChatTool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var input []map[string]any

	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, err
	}

	var tools []ChatTool

	for _, tool := range input {
		toolType, _ := tool["type"].(string)

		/*
			Codex utilise principalement :

			{
			  "type": "function",
			  "name": "...",
			  "description": "...",
			  "parameters": {...}
			}

			ou parfois :

			{
			  "type": "function",
			  "function": {...}
			}
		*/

		if toolType != "function" {
			continue
		}

		name, _ := tool["name"].(string)
		description, _ := tool["description"].(string)

		parameters := tool["parameters"]

		if nested, ok := tool["function"].(map[string]any); ok {
			if name == "" {
				name, _ = nested["name"].(string)
			}

			if description == "" {
				description, _ = nested["description"].(string)
			}

			if parameters == nil {
				parameters = nested["parameters"]
			}
		}

		if name == "" {
			continue
		}

		tools = append(tools, ChatTool{
			Type: "function",
			Function: ChatFunction{
				Name:        name,
				Description: description,
				Parameters:  parameters,
			},
		})
	}

	return tools, nil
}

func convertToolChoice(choice any) any {
	if choice == nil {
		return nil
	}

	/*
		Responses peut envoyer :

		"auto"
		"none"
		"required"

		Ces valeurs sont également comprises par
		Chat Completions.
	*/

	if str, ok := choice.(string); ok {
		return str
	}

	return choice
}

/*
	Convertit l'input Responses en messages Chat Completions.

	C'est ici que le contexte des appels d'outils
	est conservé.
*/
func convertInput(req ResponsesRequest) ([]ChatMessage, error) {
	var messages []ChatMessage

	if len(req.Instructions) > 0 {
		text := contentToText(req.Instructions)

		if strings.TrimSpace(text) != "" {
			messages = append(messages, ChatMessage{
				Role:    "developer",
				Content: text,
			})
		}
	}

	if len(req.Input) == 0 {
		return messages, nil
	}

	var inputString string

	if err := json.Unmarshal(req.Input, &inputString); err == nil {
		messages = append(messages, ChatMessage{
			Role:    "user",
			Content: inputString,
		})

		return messages, nil
	}

	var inputItems []json.RawMessage

	if err := json.Unmarshal(req.Input, &inputItems); err != nil {
		return nil, fmt.Errorf(
			"unsupported Responses input format: %w",
			err,
		)
	}

	for _, raw := range inputItems {
		var item map[string]any

		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, fmt.Errorf(
				"invalid input item: %w",
				err,
			)
		}

		itemType, _ := item["type"].(string)

		/*
			------------------------------------------------
			function_call
			------------------------------------------------
		*/

		if itemType == "function_call" {
			callID, _ := item["call_id"].(string)

			if callID == "" {
				callID, _ = item["id"].(string)
			}

			name, _ := item["name"].(string)
			arguments, _ := item["arguments"].(string)

			messages = append(messages, ChatMessage{
				Role: "assistant",
				ToolCalls: []ChatToolCall{
					{
						ID:   callID,
						Type: "function",
						Function: ChatCallFunc{
							Name:      name,
							Arguments: arguments,
						},
					},
				},
			})

			continue
		}

		/*
			------------------------------------------------
			function_call_output
			------------------------------------------------
		*/

		if itemType == "function_call_output" {
			callID, _ := item["call_id"].(string)

			output := item["output"]

			var content string

			switch value := output.(type) {
			case string:
				content = value

			default:
				data, _ := json.Marshal(value)
				content = string(data)
			}

			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    content,
				ToolCallID: callID,
			})

			continue
		}

		/*
			------------------------------------------------
			message
			------------------------------------------------
		*/

		role := "user"

		if value, ok := item["role"].(string); ok {
			role = normalizeRole(value)
		}

		content := ""

		if value, exists := item["content"]; exists {
			rawContent, err := json.Marshal(value)
			if err != nil {
				return nil, err
			}

			content = contentToText(rawContent)
		}

		if content == "" {
			if text, ok := item["text"].(string); ok {
				content = text
			}
		}

		if content == "" {
			if text, ok := item["output_text"].(string); ok {
				content = text
			}
		}

		/*
			Si c'est un message assistant avec des
			tool_calls imbriqués, on les conserve.
		*/

		if role == "assistant" {
			if calls := extractInputToolCalls(item); len(calls) > 0 {
				messages = append(messages, ChatMessage{
					Role:      "assistant",
					Content:   content,
					ToolCalls: calls,
				})

				continue
			}
		}

		if content == "" {
			data, _ := json.Marshal(item)
			content = string(data)
		}

		messages = append(messages, ChatMessage{
			Role:    role,
			Content: content,
		})
	}

	return messages, nil
}

func extractInputToolCalls(item map[string]any) []ChatToolCall {
	var result []ChatToolCall

	rawCalls, ok := item["tool_calls"].([]any)
	if !ok {
		return nil
	}

	for _, raw := range rawCalls {
		call, ok := raw.(map[string]any)
		if !ok {
			continue
		}

		id, _ := call["id"].(string)
		callType, _ := call["type"].(string)

		if callType == "" {
			callType = "function"
		}

		function, _ := call["function"].(map[string]any)

		name, _ := function["name"].(string)
		arguments, _ := function["arguments"].(string)

		result = append(result, ChatToolCall{
			ID:   id,
			Type: callType,
			Function: ChatCallFunc{
				Name:      name,
				Arguments: arguments,
			},
		})
	}

	return result
}

func normalizeRole(role string) string {
	switch role {
	case "system":
		return "system"

	case "developer":
		return "developer"

	case "assistant":
		return "assistant"

	case "tool":
		return "tool"

	case "user":
		return "user"

	default:
		return "user"
	}
}

func contentToText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var text string

	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	var parts []any

	if err := json.Unmarshal(raw, &parts); err == nil {
		var result []string

		for _, part := range parts {
			if obj, ok := part.(map[string]any); ok {

				if text, ok := obj["text"].(string); ok {
					result = append(result, text)
					continue
				}

				if text, ok := obj["content"].(string); ok {
					result = append(result, text)
					continue
				}

				/*
					Ne pas transformer function_call en
					texte si possible.
				*/
				if objType, ok := obj["type"].(string); ok {
					if objType == "function_call" ||
						objType == "function_call_output" {
						continue
					}
				}

				data, _ := json.Marshal(obj)

				if len(data) > 0 {
					result = append(result, string(data))
				}
			} else if str, ok := part.(string); ok {
				result = append(result, str)
			}
		}

		return strings.Join(result, "\n")
	}

	var value any

	if err := json.Unmarshal(raw, &value); err == nil {
		data, _ := json.Marshal(value)
		return string(data)
	}

	return string(raw)
}

/*
	Extrait le texte assistant NVIDIA.
*/
func extractAssistantText(completion map[string]any) string {
	choices, ok := completion["choices"].([]any)

	if !ok || len(choices) == 0 {
		return ""
	}

	choice, ok := choices[0].(map[string]any)

	if !ok {
		return ""
	}

	message, ok := choice["message"].(map[string]any)

	if !ok {
		return ""
	}

	switch content := message["content"].(type) {
	case string:
		return content

	case []any:
		var parts []string

		for _, item := range content {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}

			if text, ok := obj["text"].(string); ok {
				parts = append(parts, text)
			}
		}

		return strings.Join(parts, "")
	}

	return ""
}

/*
	Extrait les tool_calls de la réponse NVIDIA.
*/
func extractToolCalls(completion map[string]any) []ChatToolCall {
	choices, ok := completion["choices"].([]any)

	if !ok || len(choices) == 0 {
		return nil
	}

	choice, ok := choices[0].(map[string]any)

	if !ok {
		return nil
	}

	message, ok := choice["message"].(map[string]any)

	if !ok {
		return nil
	}

	rawCalls, ok := message["tool_calls"].([]any)

	if !ok {
		return nil
	}

	var calls []ChatToolCall

	for _, raw := range rawCalls {
		call, ok := raw.(map[string]any)
		if !ok {
			continue
		}

		id, _ := call["id"].(string)
		callType, _ := call["type"].(string)

		if callType == "" {
			callType = "function"
		}

		function, _ := call["function"].(map[string]any)

		if function == nil {
			continue
		}

		name, _ := function["name"].(string)
		arguments, _ := function["arguments"].(string)

		if id == "" {
			id = fmt.Sprintf("call_%d", len(calls))
		}

		calls = append(calls, ChatToolCall{
			ID:   id,
			Type: callType,
			Function: ChatCallFunc{
				Name:      name,
				Arguments: arguments,
			},
		})
	}

	return calls
}

/*
	Transforme la réponse Chat Completions en
	Responses API.
*/
func convertChatResponse(completion map[string]any) map[string]any {
	outputText := extractAssistantText(completion)
	toolCalls := extractToolCalls(completion)

	output := make([]any, 0)

	if outputText != "" {
		output = append(output, map[string]any{
			"type":   "message",
			"id":     "msg_nvidia_muse_glimmer",
			"role":   "assistant",
			"status": "completed",
			"content": []any{
				map[string]any{
					"type": "output_text",
					"text": outputText,
				},
			},
		})
	}

	for i, call := range toolCalls {
		output = append(output, map[string]any{
			"type":      "function_call",
			"id":        call.ID,
			"call_id":   call.ID,
			"name":      call.Function.Name,
			"arguments": call.Function.Arguments,
			"status":    "completed",
			"index":     i,
		})
	}

	status := "completed"

	if len(toolCalls) > 0 {
		status = "completed"
	}

	response := map[string]any{
		"id":     "resp_nvidia_muse_glimmer",
		"object": "response",
		"model":  nvidiaModel,
		"status": status,
		"output": output,
	}

	if outputText != "" {
		response["output_text"] = outputText
	}

	return response
}

/*
	SSE Responses API.

	Important :
	- message -> output_text
	- tool_calls -> function_call
*/
func writeResponsesSSE(
	w http.ResponseWriter,
	text string,
	toolCalls []ChatToolCall,
) {
	flusher, ok := w.(http.Flusher)

	if !ok {
		http.Error(
			w,
			"streaming not supported",
			http.StatusInternalServerError,
		)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	responseID := fmt.Sprintf(
		"resp_nvidia_muse_%d",
		time.Now().UnixNano(),
	)

	sequence := 0

	/*
		response.created
	*/

	created := map[string]any{
		"type":           "response.created",
		"sequence_number": sequence,
		"response": map[string]any{
			"id":     responseID,
			"object": "response",
			"status": "in_progress",
			"model":  nvidiaModel,
			"output": []any{},
		},
	}

	writeSSEEvent(w, "response.created", created)
	flusher.Flush()

	sequence++

	/*
		Message texte.
	*/

	if text != "" {
		itemID := fmt.Sprintf(
			"msg_nvidia_muse_%d",
			time.Now().UnixNano(),
		)

		added := map[string]any{
			"type":           "response.output_item.added",
			"sequence_number": sequence,
			"output_index":   0,
			"item": map[string]any{
				"id":      itemID,
				"type":    "message",
				"role":    "assistant",
				"status":  "in_progress",
				"content": []any{},
			},
		}

		writeSSEEvent(
			w,
			"response.output_item.added",
			added,
		)
		flusher.Flush()

		sequence++

		part := map[string]any{
			"type":           "response.content_part.added",
			"sequence_number": sequence,
			"item_id":        itemID,
			"output_index":   0,
			"content_index":  0,
			"part": map[string]any{
				"type": "output_text",
				"text": "",
			},
		}

		writeSSEEvent(
			w,
			"response.content_part.added",
			part,
		)
		flusher.Flush()

		sequence++

		delta := map[string]any{
			"type":           "response.output_text.delta",
			"sequence_number": sequence,
			"item_id":        itemID,
			"output_index":   0,
			"content_index":  0,
			"delta":          text,
		}

		writeSSEEvent(
			w,
			"response.output_text.delta",
			delta,
		)
		flusher.Flush()

		sequence++

		textDone := map[string]any{
			"type":           "response.output_text.done",
			"sequence_number": sequence,
			"item_id":        itemID,
			"output_index":   0,
			"content_index":  0,
			"text":           text,
		}

		writeSSEEvent(
			w,
			"response.output_text.done",
			textDone,
		)
		flusher.Flush()

		sequence++

		partDone := map[string]any{
			"type":           "response.content_part.done",
			"sequence_number": sequence,
			"item_id":        itemID,
			"output_index":   0,
			"content_index": 0,
			"part": map[string]any{
				"type": "output_text",
				"text": text,
			},
		}

		writeSSEEvent(
			w,
			"response.content_part.done",
			partDone,
		)
		flusher.Flush()

		sequence++

		itemDone := map[string]any{
			"type":           "response.output_item.done",
			"sequence_number": sequence,
			"output_index":   0,
			"item": map[string]any{
				"id":      itemID,
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": text,
					},
				},
			},
		}

		writeSSEEvent(
			w,
			"response.output_item.done",
			itemDone,
		)
		flusher.Flush()

		sequence++
	}

	/*
		Tool calls.
	*/

	for index, call := range toolCalls {
		itemID := call.ID

		if itemID == "" {
			itemID = fmt.Sprintf(
				"fc_%d_%d",
				time.Now().UnixNano(),
				index,
			)
		}

		callID := call.ID

		/*
			response.output_item.added
		*/

		added := map[string]any{
			"type":           "response.output_item.added",
			"sequence_number": sequence,
			"output_index":   index,
			"item": map[string]any{
				"id":        itemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   callID,
				"name":      call.Function.Name,
				"arguments": "",
			},
		}

		writeSSEEvent(
			w,
			"response.output_item.added",
			added,
		)
		flusher.Flush()

		sequence++

		/*
			arguments delta
		*/

		if call.Function.Arguments != "" {
			delta := map[string]any{
				"type":           "response.function_call_arguments.delta",
				"sequence_number": sequence,
				"item_id":        itemID,
				"output_index":   index,
				"delta":          call.Function.Arguments,
			}

			writeSSEEvent(
				w,
				"response.function_call_arguments.delta",
				delta,
			)
			flusher.Flush()

			sequence++
		}

		/*
			arguments done
		*/

		done := map[string]any{
			"type":           "response.function_call_arguments.done",
			"sequence_number": sequence,
			"item_id":        itemID,
			"output_index":   index,
			"arguments":      call.Function.Arguments,
		}

		writeSSEEvent(
			w,
			"response.function_call_arguments.done",
			done,
		)
		flusher.Flush()

		sequence++

		/*
			output_item.done
		*/

		itemDone := map[string]any{
			"type":           "response.output_item.done",
			"sequence_number": sequence,
			"output_index":   index,
			"item": map[string]any{
				"id":        itemID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   callID,
				"name":      call.Function.Name,
				"arguments": call.Function.Arguments,
			},
		}

		writeSSEEvent(
			w,
			"response.output_item.done",
			itemDone,
		)
		flusher.Flush()

		sequence++
	}

	/*
		response.completed
	*/

	completedOutput := make([]any, 0)

	if text != "" {
		completedOutput = append(
			completedOutput,
			map[string]any{
				"id":      "msg_nvidia_muse_glimmer",
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": text,
					},
				},
			},
		)
	}

	for _, call := range toolCalls {
		completedOutput = append(
			completedOutput,
			map[string]any{
				"id":        call.ID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   call.ID,
				"name":      call.Function.Name,
				"arguments": call.Function.Arguments,
			},
		)
	}

	completed := map[string]any{
		"type":           "response.completed",
		"sequence_number": sequence,
		"response": map[string]any{
			"id":     responseID,
			"object": "response",
			"status": "completed",
			"model":  nvidiaModel,
			"output": completedOutput,
		},
	}

	if text != "" {
		completedResponse := completed["response"].(map[string]any)
		completedResponse["output_text"] = text
	}

	writeSSEEvent(
		w,
		"response.completed",
		completed,
	)
	flusher.Flush()
}