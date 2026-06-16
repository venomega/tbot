package main

type Chunk struct {
	Path    string   `json:"path"`
	Start   int      `json:"start"`
	End     int      `json:"end"`
	Type    string   `json:"type"`
	Name    string   `json:"name"`
	Parent  string   `json:"parent"`
	Imports []string `json:"imports"`
	Symbols []string `json:"symbols"`
	Lang    string   `json:"lang"`
	Text    string   `json:"text"`
}

type Index struct {
	Version    int              `json:"version"`
	RootPath   string           `json:"root_path"`
	Chunks     []*Chunk         `json:"chunks"`
	BM25Params BM25Params       `json:"bm25_params"`
	Files      map[string]int64 `json:"files"`
}

type BM25Params struct {
	K1          float64            `json:"k1"`
	B           float64            `json:"b"`
	Avgdl       float64            `json:"avgdl"`
	TotalChunks int                `json:"total_chunks"`
	IDF         map[string]float64 `json:"idf"`
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
}

type chunkData struct {
	Chunk  *Chunk
	Tokens []string
	Freqs  map[string]int
}
