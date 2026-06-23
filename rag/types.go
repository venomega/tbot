package main

// maxFreq is the maximum term frequency we can pack into a uint32
// with the lower 16 bits for term index and upper 16 bits for frequency.
const maxPackedFreq = 65535

// packFreq packs a term index and frequency into a single uint32.
// term index in lower 16 bits, frequency in upper 16 bits.
func packFreq(termIdx, freq uint32) uint32 {
	return (freq << 16) | termIdx
}

func unpackFreq(packed uint32) (termIdx, freq uint32) {
	return packed & 0xFFFF, packed >> 16
}

type Chunk struct {
	Path        string         `json:"path"`
	Start       int            `json:"start"`
	End         int            `json:"end"`
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Parent      string         `json:"parent"`
	Imports     []string       `json:"imports"`
	Symbols     []string       `json:"symbols"`
	Lang        string         `json:"lang"`
	Text        string         `json:"text"`
	Freqs       map[string]int `json:"-"` // pre-computed term frequencies (v3 compat, skip JSON)
	PackedFreqs []uint32       `json:"-"` // packed term frequencies: (freq<<16)|termIdx (v4, skip JSON)
	NumTokens   int            `json:"-"` // total token count for this chunk
}

type Index struct {
	Version    int              `json:"version"`
	RootPath   string           `json:"root_path"`
	Chunks     []*Chunk         `json:"chunks"`
	BM25Params BM25Params       `json:"bm25_params"`
	Files      map[string]int64 `json:"files"`
	TermIndex  map[string]uint32 `json:"-"` // global term → numeric ID (skip JSON)
	Inverted   map[string][]int  `json:"-"` // term → list of chunk indices containing it (skip JSON)
}

type BM25Params struct {
	K1          float64            `json:"k1"`
	B           float64            `json:"b"`
	Avgdl       float64            `json:"avgdl"`
	TotalChunks int                `json:"total_chunks"`
	IDF         map[string]float64 `json:"idf"`
	IDFValues   []float64          `json:"-"` // indexed by term ID (skip JSON), built during indexing
}

type SearchResult struct {
	Path    string   `json:"path"`
	Start   int      `json:"start"`
	End     int      `json:"end"`
	Type    string   `json:"type"`
	Name    string   `json:"name"`
	Parent  string   `json:"parent"`
	Imports []string `json:"imports"`
	Symbols []string `json:"symbols"`
	Lang    string   `json:"lang"`
	Score   float64  `json:"score"`
	Text    string   `json:"text"`
	Snippet string   `json:"snippet,omitempty"`
}

type chunkData struct {
	Chunk  *Chunk
	Tokens []string
	Freqs  map[string]int
}
