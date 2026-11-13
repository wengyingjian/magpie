package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// ebaoCap is a custom provider's gateway turning away a reply limit over
// the 128000 it takes, in FastAPI's own words: models.dev lists the model
// at 384000, magpie published that to DeepSeek Harness, and every turn
// asking it was refused "body -> max_tokens: Input should be less than or
// equal to 128000" (reported 2026-11-13).
const ebaoCap = `{"detail":[{"type":"less_than_equal","loc":["body","max_tokens"],"msg":"Input should be less than or equal to 128000","input":384000,"ctx":{"le":128000}}]}`

const ebaoMost = 128000

// replyCapped is an OpenAI-compatible upstream that answers a max_tokens
// over most with ebaoCap, for the model named only (for every one, when it
// names none), and answers otherwise.
type replyCapped struct {
	most int
	name string
	mu   sync.Mutex
	got  [][]byte
}

func (c *replyCapped) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	c.mu.Lock()
	c.got = append(c.got, b)
	c.mu.Unlock()
	var q struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
	}
	json.Unmarshal(b, &q)
	w.Header().Set("Content-Type", "application/json")
	if q.MaxTokens > c.most && (c.name == "" || strings.HasSuffix(q.Model, c.name)) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		io.WriteString(w, ebaoCap)
		return
	}
	io.WriteString(w, `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
}

func (c *replyCapped) calls() [][]byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([][]byte(nil), c.got...)
}

// askedMax is the reply limit of a body the upstream was sent.
func askedMax(t *testing.T, body []byte) int {
	t.Helper()
	var q struct {
		MaxTokens int `json:"max_tokens"`
	}
	if err := json.Unmarshal(body, &q); err != nil {
		t.Fatal(err)
	}
	return q.MaxTokens
}

// A turn asking for more than a custom provider's gateway takes is turned
// away once; magpie learns the limit the refusal says and asks again
// within it, and its later turns go fitted at once. Another model of the
// same provider, which the refusal said nothing about, keeps the limit it
// asks for.
func TestReplyLimitOverVendorsAskedAgainWithinIt(t *testing.T) {
	fresh(t)
	up := &replyCapped{most: ebaoMost, name: "deepseek-v4.1-flash"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{ID: "ebao", Name: "ebao", Key: "k", Models: []string{"deepseek-v4.1-flash", "glm-5.3"}, Chat: srv.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	turn := func(model string) {
		t.Helper()
		body := `{"model":"ebao/` + model + `","messages":[{"role":"user","content":"hi"}],"max_tokens":384000}`
		if code, raw := postJSON(t, s, "/v1/chat/completions", body); code != 200 {
			t.Fatalf("%s: the turn failed: %d %.400s", model, code, raw)
		}
	}
	turn("deepseek-v4.1-flash")
	calls := up.calls()
	if len(calls) != 2 {
		t.Fatalf("asked %d times", len(calls))
	}
	if got := askedMax(t, calls[1]); got != ebaoMost {
		t.Fatalf("asked again within %d, not %d", got, ebaoMost)
	}
	// the cap is learned: the next turn goes fitted, asked once
	turn("deepseek-v4.1-flash")
	if calls = up.calls(); len(calls) != 3 {
		t.Fatalf("the next turn was asked %d times", len(calls)-2)
	}
	if got := askedMax(t, calls[2]); got != ebaoMost {
		t.Fatalf("the next turn asked %d, not fitted", got)
	}
	// a sibling model the refusal said nothing about keeps its own limit
	turn("glm-5.3")
	if calls = up.calls(); len(calls) != 4 {
		t.Fatalf("the sibling was asked %d times", len(calls)-3)
	}
	if got := askedMax(t, calls[3]); got != 384000 {
		t.Fatalf("the sibling asked %d, not 384000", got)
	}
}

// The limit a refusal names, in each vendor's words: the figure beside the
// field, past the ask when both are named, and none from a refusal that
// names no figure or doesn't name the field at all.
func TestStatedOutputCap(t *testing.T) {
	for _, c := range []struct {
		name  string
		body  string
		asked int
		want  int
	}{
		{"pydantic, the ask beside its cap", ebaoCap, 384000, ebaoMost},
		{"OpenAI, the cap before the field", `{"error":{"message":"max_tokens is too large: 200000. This model supports at most 16384 completion tokens, whereas you provided 200000 tokens.","type":"invalid_request_error"}}`, 200000, 16384},
		{"Anthropic, the ask past the cap", `{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 200000 > 64000, which is the maximum allowed number of output tokens for claude-sonnet-4-5"}}`, 200000, 64000},
		{"a range", `{"error":{"message":"the valid range of max_tokens is [1, 8192]"}}`, 16000, 8192},
		{"a window beside it", `{"error":{"message":"max_tokens 200000 exceeds the maximum allowed 64000 tokens for this model (context 128000)"}}`, 200000, 64000},
		{"no figure", `{"error":{"message":"max_tokens must be a positive integer"}}`, 16000, 0},
		{"the window's own refusal", `{"error":{"message":"This model's maximum context length is 1000000 tokens. However, you requested 1100000 tokens"}}`, 384000, 0},
	} {
		if got := statedOutputCap([]byte(c.body), c.asked); got != c.want {
			t.Errorf("%s: %d, want %d", c.name, got, c.want)
		}
	}
}
