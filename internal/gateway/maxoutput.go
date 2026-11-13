package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/tidwall/gjson"

	"github.com/yetone/magpie/internal/lazyre"
	"github.com/yetone/magpie/internal/provider"
)

// outputLimit is the most a reply of a provider's model may hold, where
// known: its catalog entry's (the user's setting, else the vendor's list
// or models.dev).
func outputLimit(p provider.Provider, model string) int {
	cat := provider.Served()
	for i := range cat {
		e := &cat[i] // an Entry is ~1.3KB, the catalog hundreds
		if e.Group == "" && e.Provider.ID == p.ID && (e.Model == model || e.ID == model) && e.Output > 0 {
			return e.Output
		}
	}
	return 0
}

// userOutput is the Max output the user set for p's model (provider ›
// model, or the provider's "*"), 0 when they set none. A request asking
// for more is lowered to it on any route, not only in a group: Claude Code
// asks claude-* models for their full 128000 and isn't told magpie's
// figure for them (claudeOutput), so a key whose quota is held per request
// as max_tokens times the price turned every turn away, and the field the
// editor says caps max_tokens did nothing (YaCo on Discord).
func userOutput(p provider.Provider, model string) int {
	cat := provider.Served()
	for i := range cat {
		e := &cat[i] // an Entry is ~1.3KB, the catalog hundreds
		if e.Group == "" && e.Provider.ID == p.ID && (e.Model == model || e.ID == model) && e.OutputSet && e.Output > 0 {
			return e.Output
		}
	}
	return 0
}

// askedOutput is the reply limit a request that names none is sent on
// Anthropic's Messages API, which needs one: the model's own as the agents
// are told it (the user's Max output, else the vendor's list or
// models.dev, within the window), or 0 when none is known. A Codex turn
// names none, and was sent a flat 16384, which a long thinking and tool
// turn ran out of, Codex showing "Incomplete response returned, reason:
// max_output_tokens" (#1792, tensornull).
func askedOutput(p provider.Provider, model string) int {
	cat := provider.Served()
	for i := range cat {
		e := &cat[i] // an Entry is ~1.3KB, the catalog hundreds
		if e.Group == "" && e.Provider.ID == p.ID && (e.Model == model || e.ID == model) {
			return e.PublishedOutput()
		}
	}
	return 0
}

// defaultMaxTokens is what a request naming no reply limit is sent on
// Anthropic's API for a model whose limit isn't known.
const defaultMaxTokens = 16384

// withMaxOutput is body, in proto's words, asking for a reply of at most
// limit tokens: a request that asks for more is lowered to limit, and
// Anthropic's thinking budget, which must be under it, with it. A request
// that asks for limit or less, or names no limit, goes as it is.
func withMaxOutput(proto provider.Protocol, body []byte, limit int) []byte {
	var v struct {
		MaxTokens           int `json:"max_tokens"`
		MaxCompletionTokens int `json:"max_completion_tokens"`
		MaxOutputTokens     int `json:"max_output_tokens"`
		Thinking            *struct {
			Type   string `json:"type"`
			Budget int    `json:"budget_tokens"`
		} `json:"thinking"`
		GenerationConfig map[string]any `json:"generationConfig"`
	}
	if limit <= 0 || json.Unmarshal(body, &v) != nil {
		return body
	}
	fields := map[string]any{}
	switch proto {
	case provider.Chat:
		if v.MaxCompletionTokens > limit {
			fields["max_completion_tokens"] = limit
		}
		if v.MaxTokens > limit {
			fields["max_tokens"] = limit
		}
	case provider.Responses:
		if v.MaxOutputTokens > limit {
			fields["max_output_tokens"] = limit
		}
	case provider.Anthropic:
		if v.MaxTokens <= limit {
			return body
		}
		fields["max_tokens"] = limit
		if t := v.Thinking; t != nil && t.Type == "enabled" && t.Budget >= limit {
			if limit-1 >= 1024 {
				fields["thinking"] = map[string]any{"type": "enabled", "budget_tokens": limit - 1}
			} else {
				// no budget fits under it (Anthropic's least is 1024)
				fields["thinking"] = map[string]any{"type": "disabled"}
			}
		}
	case provider.Gemini:
		if n, ok := v.GenerationConfig["maxOutputTokens"].(float64); ok && int(n) > limit {
			v.GenerationConfig["maxOutputTokens"] = limit
			fields["generationConfig"] = v.GenerationConfig
		}
	}
	if len(fields) == 0 {
		return body
	}
	return withFields(body, fields)
}

// replyLeft reads, from a refusal that counts the reply asked for in the
// window (DeepSeek's and vLLM's "This model's maximum context length is
// 1000000 tokens. However, you requested 1100000 tokens (706784 in the
// messages, 393216 in the completion)"), the tokens the window has left
// for a reply after the prompt; 0 when the error isn't one of those, or
// the prompt alone fills the window.
func replyLeft(msg []byte) int {
	window, prompt := 0, 0
	if m := windowAndPrompt.FindSubmatch(msg); m != nil {
		window, _ = strconv.Atoi(string(m[1]))
		prompt, _ = strconv.Atoi(string(m[2]))
		if m[2] == nil {
			prompt, _ = strconv.Atoi(string(m[3]))
		}
	} else if m := inputPlusReply.FindSubmatch(msg); m != nil {
		prompt, _ = strconv.Atoi(string(m[1]))
		window, _ = strconv.Atoi(string(m[2]))
	} else if m := routerWindow.FindSubmatch(msg); m != nil {
		window, _ = strconv.Atoi(string(m[1]))
		for _, part := range routerInput.FindAllSubmatch(m[2], -1) {
			n, _ := strconv.Atoi(string(part[1]))
			prompt += n
		}
	}
	if window <= 0 || prompt <= 0 || prompt >= window {
		return 0
	}
	return window - prompt
}

// windowAndPrompt matches the window and the prompt's part of a refusal
// that adds the reply's allowance to the prompt: "(N in the messages, M in
// the completion)", or vLLM's newer "your prompt contains at least N input
// tokens".
var windowAndPrompt = lazyre.New(`(?is)maximum context length is (\d+) tokens.{0,200}?(?:\((\d+) in the messages, \d+ in the completion\)|your prompt contains at least (\d+) input tokens)`)

// routerWindow matches OpenRouter's refusal, which splits the prompt by
// kind and counts the reply in the window: "This endpoint's maximum
// context length is 1048576 tokens. However, you requested about 1050086
// tokens (101187 of text input, 5181 of tool input, 943718 in the output)".
// Group 2 is the parenthesis; routerInput reads each "N of <kind> input".
var routerWindow = lazyre.New(`(?is)maximum context length is (\d+) tokens\..{0,100}?requested about \d+ tokens \(([^)]{0,200}?\d+ in the output)\)`)
var routerInput = lazyre.New(`(?i)(\d+) of [a-z ]{1,20}? input`)

// inputPlusReply is Anthropic's words for the same, which an endpoint
// speaking its API may answer with: "input length and `max_tokens` exceed
// context limit: 197202 + 21333 > 200000".
var inputPlusReply = lazyre.New(`(?i)input length and .{0,3}max_tokens.{0,3} exceed context limit:\s*(\d+)\s*\+\s*\d+\s*>\s*(\d+)`)

// roomedOutput is the reply limit n a request is filled in with, lowered
// to what a window holds with a prompt of about promptTokens: the model's
// own limit may be nearly the whole window (OpenRouter lists 943718 for a
// DeepSeek model of 1048576), and a vendor that counts the reply in the
// window refuses the pair (H1_B1_K1 on X). The prompt is sized from its
// bytes, rounded up; room under leastReplyLeft leaves n as it was, and the
// refusal tells the agent to compact. An unknown window leaves n alone.
func roomedOutput(n, window, promptTokens int) int {
	if n <= 0 || window <= 0 {
		return n
	}
	room := window - promptTokens*5/4 - 1024
	if room >= leastReplyLeft && room < n {
		return room
	}
	return n
}

// contextOf is the context window of p's model, 0 when unknown.
func contextOf(p provider.Provider, model string) int {
	cat := provider.Served()
	for i := range cat {
		e := &cat[i]
		if e.Group == "" && e.Provider.ID == p.ID && (e.Model == model || e.ID == model) {
			return e.Context
		}
	}
	return 0
}

// leastReplyLeft is the least room left for a reply that is worth asking
// for: less, and the conversation needs compacting, which the refusal
// tells the agent as it is (DeepSeek's own default reply is 8K).
const leastReplyLeft = 8192

// withReplyCap is body asking for a reply of at most limit tokens, in
// proto's words, whether or not it named a limit before: a request that
// named none was given the vendor's own, which was too large. false when
// it asked for no more already, or proto keeps no such field magpie sets.
func withReplyCap(proto provider.Protocol, body []byte, limit int) ([]byte, bool) {
	if limit <= 0 {
		return nil, false
	}
	out := withMaxOutput(proto, body, limit)
	if !bytes.Equal(out, body) {
		return out, true
	}
	var key string
	switch proto {
	case provider.Chat:
		if gjson.GetBytes(body, "max_tokens").Exists() || gjson.GetBytes(body, "max_completion_tokens").Exists() {
			return nil, false
		}
		key = "max_tokens"
	case provider.Responses:
		if gjson.GetBytes(body, "max_output_tokens").Exists() {
			return nil, false
		}
		key = "max_output_tokens"
	case provider.Gemini:
		gc := gjson.GetBytes(body, "generationConfig")
		if gc.Get("maxOutputTokens").Exists() {
			return nil, false
		}
		m := map[string]any{}
		if gc.IsObject() {
			dec := json.NewDecoder(strings.NewReader(gc.Raw))
			dec.UseNumber()
			if dec.Decode(&m) != nil {
				return nil, false
			}
		}
		m["maxOutputTokens"] = limit
		return withFields(body, map[string]any{"generationConfig": m}), true
	default:
		// Anthropic's always names one
		return nil, false
	}
	return withFields(body, map[string]any{key: limit}), true
}

// namesOutput is whether body, in proto's words, names a reply limit.
func namesOutput(proto provider.Protocol, body []byte) bool {
	switch proto {
	case provider.Chat:
		return gjson.GetBytes(body, "max_tokens").Exists() || gjson.GetBytes(body, "max_completion_tokens").Exists()
	case provider.Responses:
		return gjson.GetBytes(body, "max_output_tokens").Exists()
	case provider.Anthropic:
		return gjson.GetBytes(body, "max_tokens").Exists()
	case provider.Gemini:
		return gjson.GetBytes(body, "generationConfig.maxOutputTokens").Exists()
	}
	return true
}

// shortByDefault is whether p's API cuts a reply that names no limit far
// short of what its model can give: DeepSeek's gives 8K without thinking
// and 64K with it (api-docs.deepseek.com, create-chat-completion:
// "When not set, the default is 8K in non-thinking mode, 64K in thinking
// mode"), on /v1/responses too, where its models go to 384K. Codex names
// no max_output_tokens, so its turns there stopped at 8192 tokens, "Incomplete
// response returned, reason: max_output_tokens" (xuanwo on X). Checked
// against api.deepseek.com on 2026-10-27: deepseek-flash, effort none, no
// limit: incomplete at 8192; max_output_tokens 393216: completed at 17602.
func shortByDefault(p provider.Provider) bool {
	return p.VendorHost() == "api.deepseek.com"
}

// withOwnOutput is body, in proto's words, asking p's model for the reply
// its own limit allows (askedOutput) when it names none and p's API would
// otherwise cut it short (shortByDefault). A limit the client named goes
// as it is.
func withOwnOutput(p provider.Provider, model string, proto provider.Protocol, body []byte) []byte {
	if !shortByDefault(p) || namesOutput(proto, body) {
		return body
	}
	n := roomedOutput(askedOutput(p, model), contextOf(p, model), len(body)/4)
	if n <= 0 {
		return body
	}
	if b, ok := withReplyCap(proto, body, n); ok {
		return b
	}
	return body
}

// fillsOutput is whether a request translated for p's API on to that
// names no reply limit is sent its model's own (askedOutput): the client's
// API reads none as the model's whole, which another vendor's needn't
// (DeepSeek's reads it as 8K, shortByDefault). Anthropic's always is, as
// it needs one. An account's own backend (the ChatGPT one turns
// max_output_tokens away) isn't. Another magpie is: this fills only a
// translated turn (forwardTranslated), and one translated here goes to
// the other magpie with its tools converted, a body it passes whole —
// its own filling is translation work it never does (#5566), so its
// vendor's short default was all a cascade got.
func fillsOutput(p provider.Provider, to provider.Protocol) bool {
	if to == provider.Anthropic {
		return true
	}
	return (to == provider.Chat || to == provider.Responses) &&
		(p.Account == nil || p.IsPlugin())
}

// outputFieldRefused is a request error about the reply limit itself
// ("max_tokens must be at most 32768"), not the window it shares with the
// prompt, which replyLeft reads and the request is asked again for.
func outputFieldRefused(status int, body []byte) bool {
	if !badRequest(status) {
		return false
	}
	w := wordsOf(body)
	if replyLeft(w) > 0 {
		return false
	}
	m := strings.ToLower(string(w))
	return strings.Contains(m, "max_tokens") || strings.Contains(m, "max_output_tokens") ||
		strings.Contains(m, "max_completion_tokens") || strings.Contains(m, "maxoutputtokens")
}

// outputFillRefused names, for markUnfit, a model whose vendor turned
// away the reply limit magpie filled in: a request naming none is sent
// none there from then on, as before.
func outputFillRefused(model string) string { return "filled-output:" + model }

// bodyOutput is the reply limit body, in proto's words, asks for: 0 when
// it names none.
func bodyOutput(proto provider.Protocol, body []byte) int {
	switch proto {
	case provider.Chat:
		return max(int(gjson.GetBytes(body, "max_tokens").Int()), int(gjson.GetBytes(body, "max_completion_tokens").Int()))
	case provider.Responses:
		return int(gjson.GetBytes(body, "max_output_tokens").Int())
	case provider.Anthropic:
		return int(gjson.GetBytes(body, "max_tokens").Int())
	case provider.Gemini:
		return int(gjson.GetBytes(body, "generationConfig.maxOutputTokens").Int())
	}
	return 0
}

// outputCapWords are a vendor's words for the reply limit's field, whose
// cap it may name beside them.
var outputCapWords = lazyre.New(`(?i)max_output_tokens|max_completion_tokens|max_tokens|maxoutputtokens|max (output |completion )?tokens`)

// outputCapDigits are the figures a refusal names.
var outputCapDigits = lazyre.New(`\d{3,9}`)

// outputCapSlack is how far from a mention of the reply limit's field its
// cap may sit: pydantic names the field a sentence before its "less than
// or equal to N", and OpenAI's "supports at most N completion tokens"
// names it after the figure.
const outputCapSlack = 120

// leastOutputCap is the smallest figure read as a reply limit: a refusal
// that names the range it takes ("the valid range of max_tokens is
// [1, 8192]") names the least beside the cap.
const leastOutputCap = 512

// statedOutputCap is the reply limit a refusal says is the most it takes:
// the smallest figure beside a mention of the limit's field that a reply
// could hold and that the request asks more than (a refusal often names
// the ask too: Anthropic's "max_tokens: 200000 > 64000, which is the
// maximum allowed"). 0 when it names none.
func statedOutputCap(errBody []byte, asked int) int {
	if asked <= 0 {
		return 0
	}
	w := wordsOf(errBody)
	least := 0
	for _, m := range outputCapWords.FindAllIndex(w, -1) {
		n := leastFigure(w[m[1]:min(len(w), m[1]+outputCapSlack)], asked)
		if n == 0 {
			n = leastFigure(w[max(0, m[1]-outputCapSlack):m[1]], asked)
		}
		if n > 0 && (least == 0 || n < least) {
			least = n
		}
	}
	return least
}

// leastFigure is the smallest number in w that a reply could hold and that
// the request asks more than: 0 when there is none.
func leastFigure(w []byte, asked int) int {
	least := 0
	for _, m := range outputCapDigits.FindAll(w, -1) {
		n, err := strconv.Atoi(string(m))
		if err != nil || n < leastOutputCap || n >= asked {
			continue
		}
		if least == 0 || n < least {
			least = n
		}
	}
	return least
}

// outputCaps are the reply limits providers said were the most they take,
// by provider id and model, for as long as magpie runs. A vendor's figure
// in a refusal is the one reply limit that is what it takes now: models.dev's
// figure for a custom provider's model can be the model's, while the gateway
// in front of it takes less (ebao's 128000 for deepseek-v4.1-flash, which
// models.dev lists at 384000), and magpie published that larger one to the
// agent.
type outputCaps struct {
	mu sync.Mutex
	m  map[string]int
}

func (c *outputCaps) of(providerID, model string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.m[providerID+"\x00"+model]
}

// learn keeps the smaller of n and the limit known before.
func (c *outputCaps) learn(providerID, model string, n int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[string]int{}
	}
	k := providerID + "\x00" + model
	if old := c.m[k]; old > 0 && old <= n {
		return old
	}
	c.m[k] = n
	return n
}

// withinOutputCap is body as it goes to p: its reply limit lowered to the
// one p's vendor said was the most it takes, when it turned a larger one
// away before (outputCaps).
func (s *Server) withinOutputCap(p provider.Provider, to provider.Protocol, model, path string, body []byte) []byte {
	if countsTokens(path) {
		return body
	}
	n := s.outputCaps.of(p.ID, model)
	if n <= 0 {
		return body
	}
	return withMaxOutput(to, body, n)
}

// refitOverOutput is forward's answer to res, a refusal of the reply limit
// body asks for (outputFieldRefused): the limit the vendor says it takes
// is learned for p's model, and the request asked again within it, so a
// turn asking for more than a custom provider's gateway takes is answered
// instead of failed every time it comes.
func (s *Server) refitOverOutput(ctx context.Context, p provider.Provider, to provider.Protocol, model, path string, body []byte, in http.Header, res *http.Response) (*http.Response, error) {
	for range 2 {
		if res.StatusCode < 400 || countsTokens(path) {
			return res, nil
		}
		b := peekError(res)
		asked := bodyOutput(to, body)
		if asked <= 0 || !outputFieldRefused(res.StatusCode, b) {
			return res, nil
		}
		n := statedOutputCap(b, asked)
		if n <= 0 {
			return res, nil
		}
		if n = s.outputCaps.learn(p.ID, model, n); n >= asked {
			return res, nil
		}
		log.Printf("%s: %s takes a reply of at most %d tokens, and %d was asked; asking again within it", p.Name, model, n, asked)
		body = withMaxOutput(to, body, n)
		res.Body.Close()
		var err error
		if res, err = s.forwardOnce(ctx, p, to, path, body, in); err != nil {
			return res, err
		}
	}
	return res, nil
}

// thinksByDefault is whether p's API runs a model's thinking on a request
// that names none, and counts it in the reply limit: Tencent Cloud's
// TokenHub and Token Plan serve DeepSeek with thinking "enabled" and
// reasoning_effort "high" unless the request says otherwise, and advise
// max_tokens of at least 2048 in thinking mode (cloud.tencent.com/document/
// product/1823/132248). A memory tool's max_tokens of 4096 was spent on
// reasoning alone, every reply cut at its length with no answer (WaiooT on X).
func thinksByDefault(p provider.Provider, model string) bool {
	return strings.HasSuffix(p.VendorHost(), "tencentmaas.com") && strings.Contains(strings.ToLower(model), "deepseek")
}

// namesReasoning is whether body, in proto's words, says how the model is
// to reason (or not to): the client's own choice, which goes as it is.
func namesReasoning(proto provider.Protocol, body []byte) bool {
	switch proto {
	case provider.Chat:
		for _, k := range []string{"reasoning_effort", "thinking", "enable_thinking", "chat_template_kwargs", "reasoning"} {
			if gjson.GetBytes(body, k).Exists() {
				return true
			}
		}
		return false
	case provider.Responses:
		return gjson.GetBytes(body, "reasoning").Exists()
	}
	return true
}

// withThinkingRoom is body, in proto's words, with the reply limit a client
// named raised by the room a default-thinking model (thinksByDefault) needs
// to think in before it answers (replyRoom at its default effort, high),
// when the client asked for no reasoning of its own, the user set no Max
// output for the model, and the limit is a small one (at most
// thinkingRoomBelow). The model's own limit, where known, still caps it.
func withThinkingRoom(p provider.Provider, model string, proto provider.Protocol, body []byte) []byte {
	if !thinksByDefault(p, model) || namesReasoning(proto, body) || userOutput(p, model) > 0 {
		return body
	}
	var field string
	switch proto {
	case provider.Chat:
		field = "max_tokens"
		if !gjson.GetBytes(body, field).Exists() {
			field = "max_completion_tokens"
		}
	case provider.Responses:
		field = "max_output_tokens"
	default:
		return body
	}
	n := int(gjson.GetBytes(body, field).Int())
	if n <= 0 || n > thinkingRoomBelow {
		return body
	}
	n = replyRoom(n, "high")
	if own := askedOutput(p, model); own > 0 && n > own {
		n = own
	}
	return withFields(body, map[string]any{field: n})
}

// thinkingRoomBelow is the largest reply limit a request names that is
// raised for thinking: a client that asked for more has made room.
const thinkingRoomBelow = 16384
