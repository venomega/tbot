package main

import (
	"math"
	"sort"
	"strings"
)

const (
	defaultK1 = 1.5
	defaultB  = 0.75
)

type docFreq struct {
	Freqs      map[string]int
	TotalTerms int
}

func buildBM25(chunks []*chunkData, k1, b float64) BM25Params {
	n := len(chunks)
	if n == 0 {
		return BM25Params{K1: k1, B: b}
	}

	var totalTerms int
	df := make(map[string]int)
	for _, cd := range chunks {
		for term := range cd.Freqs {
			df[term]++
		}
		totalTerms += len(cd.Tokens)
	}

	avgdl := float64(totalTerms) / float64(n)
	idf := make(map[string]float64, len(df))
	for term, d := range df {
		idf[term] = math.Log(1 + (float64(n)-float64(d)+0.5)/(float64(d)+0.5))
	}

	return BM25Params{
		K1:          k1,
		B:           b,
		Avgdl:       avgdl,
		TotalChunks: n,
		IDF:         idf,
	}
}

// scoreChunkString is the original string-map-based scorer (used for legacy index fallback).
func scoreChunkString(queryTokens []string, freqs map[string]int, numTerms int, p BM25Params) float64 {
	var score float64
	for _, qt := range queryTokens {
		idf, ok := p.IDF[qt]
		if !ok {
			continue
		}
		f := float64(freqs[qt])
		docLen := float64(numTerms)
		score += idf * (f * (p.K1 + 1)) / (f + p.K1*(1-p.B+p.B*docLen/p.Avgdl))
	}
	return score
}

// scoreChunkPacked scores a chunk using packed term frequencies (termIdx<<16 | freq).
func scoreChunkPacked(querySet map[uint32]struct{}, packed []uint32, numTokens int, p BM25Params) float64 {
	var score float64
	docLen := float64(numTokens)
	for _, entry := range packed {
		termIdx, freq := unpackFreq(entry)
		if _, ok := querySet[termIdx]; !ok {
			continue
		}
		idf := p.IDFValues[termIdx]
		ff := float64(freq)
		score += idf * (ff * (p.K1 + 1)) / (ff + p.K1*(1-p.B+p.B*docLen/p.Avgdl))
	}
	return score
}

// ── Field weights ──
// Multiplicadores de score según dónde matchea el query en el chunk.
// Sin coste adicional de indexación, se calculan en tiempo de búsqueda.

func fieldWeight(chunk *Chunk, queryTokens []string) float64 {
	weight := 1.0
	if chunk == nil || len(queryTokens) == 0 {
		return weight
	}

	for _, qt := range queryTokens {
		// Match en nombre de función/clase/struct: peso 2.5x
		if strings.Contains(strings.ToLower(chunk.Name), qt) {
			weight += 1.5
		}
		// Match en type (struct, interface, function, method): peso 1.3x
		if strings.Contains(strings.ToLower(chunk.Type), qt) {
			weight += 0.3
		}
		// Match en símbolos (fields, methods exportados): peso 2.0x
		for _, sym := range chunk.Symbols {
			if strings.Contains(strings.ToLower(sym), qt) {
				weight += 1.0
				break
			}
		}
		// Match en imports: peso 1.5x
		for _, imp := range chunk.Imports {
			if strings.Contains(strings.ToLower(imp), qt) {
				weight += 0.5
				break
			}
		}
		// Match en parent (clase/struct contenedor): peso 2.0x
		if strings.Contains(strings.ToLower(chunk.Parent), qt) {
			weight += 1.0
		}
	}

	return weight
}

// ── Query expansion (sinónimos español) ──
// Expande términos de búsqueda con sinónimos comunes.
// Se aplica UNA VEZ por query, no por chunk. Coste negligible.

var spanishSynonyms = map[string][]string{
	// Verbos comunes
	"crear":   {"generar", "construir", "nuevo", "hacer"},
	"buscar":  {"encontrar", "localizar", "consultar", "recuperar"},
	"obtener": {"recuperar", "conseguir", "leer", "sacar"},
	"eliminar": {"borrar", "remover", "quitar", "suprimir"},
	"actualizar": {"modificar", "editar", "cambiar", "update"},
	"insertar": {"agregar", "anadir", "incluir", "meter"},
	"listar":   {"enumerar", "mostrar", "ver", "obtener"},
	"procesar": {"ejecutar", "realizar", "correr", "manipular"},
	"configurar": {"establecer", "definir", "ajustar", "setear"},
	"validar":  {"verificar", "comprobar", "chequear"},
	"iniciar":  {"comenzar", "empezar", "arrancar", "start"},
	"finalizar": {"terminar", "acabar", "completar", "end"},
	"cargar":   {"load", "leer", "importar"},
	"guardar":  {"save", "almacenar", "persistir", "escribir"},

	// Sustantivos comunes en código
	"archivo":  {"fichero", "file", "documento"},
	"usuario":  {"user", "cliente", "persona"},
	"datos":    {"data", "informacion", "contenido"},
	"funcion":  {"function", "metodo", "operacion"},
	"variable": {"var", "campo", "atributo", "field"},
	"clase":    {"class", "tipo", "estructura"},
	"interfaz": {"interface", "contrato", "api"},
	"lista":    {"array", "slice", "coleccion", "vector"},
	"mapa":     {"map", "diccionario", "hash"},
	"error":    {"err", "fallo", "exception"},
	"prueba":   {"test", "testear", "probar", "verificar"},
	"base":     {"base de datos", "bd", "database", "db", "sql"},
	"servicio": {"service", "api", "endpoint", "rest"},

	// Inglés → español (para consultas mixtas)
	"create": {"crear", "nuevo", "generar"},
	"find":   {"buscar", "encontrar", "localizar"},
	"delete": {"eliminar", "borrar"},
	"update": {"actualizar", "modificar"},
	"get":    {"obtener", "recuperar", "leer"},
	"set":    {"establecer", "definir", "poner"},
	"save":   {"guardar", "almacenar"},
	"load":   {"cargar", "leer"},
}

func expandQuery(tokens []string) []string {
	if len(tokens) == 0 {
		return tokens
	}
	// Usar un map para deduplicar
	seen := make(map[string]struct{}, len(tokens)*2)
	result := make([]string, 0, len(tokens)*2)

	for _, t := range tokens {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		result = append(result, t)
		// Añadir sinónimos
		if syns, ok := spanishSynonyms[t]; ok {
			for _, s := range syns {
				// Tokenizar el sinónimo (puede ser multi-palabra como "base de datos")
				synTokens := tokenize(s)
				for _, st := range synTokens {
					if _, ok := seen[st]; !ok {
						seen[st] = struct{}{}
						result = append(result, st)
					}
				}
			}
		}
	}

	return result
}

// ── Snippet mejorado ──
// Busca el mejor párrafo/bloque para mostrar como contexto.
// Para código, intenta incluir la firma completa de la función.

func extractSnippet(text string, queryTokens []string, contextLines int) string {
	if text == "" {
		return ""
	}
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return ""
	}

	// Puntuar cada línea por matches de query tokens (case-insensitive)
	lineScores := make([]int, len(lines))
	maxScore := 0
	bestLine := 0
	for i, line := range lines {
		lower := strings.ToLower(line)
		score := 0
		for _, qt := range queryTokens {
			if strings.Contains(lower, qt) {
				score++
			}
		}
		lineScores[i] = score
		if score > maxScore {
			maxScore = score
			bestLine = i
		}
	}

	if maxScore == 0 {
		// Sin match directo, devolver principio del chunk
		end := contextLines + 1
		if end > len(lines) {
			end = len(lines)
		}
		return strings.TrimSpace(strings.Join(lines[:end], "\n"))
	}

	// Estrategia: expandir hacia atrás hasta un separador semántico
	// (línea en blanco, cabecera de sección, llave de apertura de función)
	start := bestLine
	for start > 0 {
		trimmed := strings.TrimSpace(lines[start-1])
		// Parar en: línea en blanco, cabecera markdown, llave de cierre
		if trimmed == "" || strings.HasPrefix(trimmed, "#") ||
			strings.HasPrefix(trimmed, "```") || trimmed == "}" || trimmed == "{" {
			break
		}
		start--
	}

	// Expandir hacia adelante hasta separador semántico
	end := bestLine
	for end < len(lines) {
		trimmed := strings.TrimSpace(lines[end])
		if trimmed == "" || strings.HasPrefix(trimmed, "#") ||
			strings.HasPrefix(trimmed, "```") {
			end++
			break
		}
		end++
	}

	// Si el bloque es muy pequeño, expandir con líneas adicionales de contexto
	if end-start < contextLines {
		extra := contextLines - (end - start)
		start2 := start - extra/2
		if start2 < 0 {
			start2 = 0
		}
		end2 := end + extra/2
		if end2 > len(lines) {
			end2 = len(lines)
		}
		start, end = start2, end2
	}

	// Asegurar límites
	if start < 0 {
		start = 0
	}
	if end > len(lines) {
		end = len(lines)
	}

	snippet := strings.Join(lines[start:end], "\n")
	return strings.TrimSpace(snippet)
}

// ── Search principal ──

func search(idx *Index, query string, topK int) []SearchResult {
	if len(idx.Chunks) == 0 {
		return nil
	}

	// Tokenizar query
	baseTokens := tokenize(query)
	if len(baseTokens) == 0 {
		return nil
	}

	// Expandir query con sinónimos (una vez, no por chunk)
	qtokens := expandQuery(baseTokens)

	// Pre-convert query tokens to term IDs for fast scoring
	var queryTermIDs []uint32
	querySet := make(map[uint32]struct{}, len(qtokens))
	if idx.TermIndex != nil {
		for _, t := range qtokens {
			if id, ok := idx.TermIndex[t]; ok {
				queryTermIDs = append(queryTermIDs, id)
				querySet[id] = struct{}{}
			}
		}
	}

	// Determinar chunks candidatos via inverted index
	var candidateIndices []int
	if idx.Inverted != nil {
		seen := make(map[int]struct{})
		for _, qt := range qtokens {
			if indices, ok := idx.Inverted[qt]; ok {
				for _, ci := range indices {
					seen[ci] = struct{}{}
				}
			}
		}
		if len(seen) == 0 {
			return nil
		}
		candidateIndices = make([]int, 0, len(seen))
		for ci := range seen {
			candidateIndices = append(candidateIndices, ci)
		}
	} else {
		candidateIndices = make([]int, len(idx.Chunks))
		for i := range idx.Chunks {
			candidateIndices[i] = i
		}
	}

	type scored struct {
		chunk *Chunk
		score float64
	}

	pre := make([]scored, 0, len(candidateIndices))
	for _, ci := range candidateIndices {
		c := idx.Chunks[ci]
		var bm25Score float64

		if c.PackedFreqs != nil && len(querySet) > 0 && len(idx.BM25Params.IDFValues) > 0 {
			bm25Score = scoreChunkPacked(querySet, c.PackedFreqs, c.NumTokens, idx.BM25Params)
		} else if c.Freqs != nil {
			bm25Score = scoreChunkString(qtokens, c.Freqs, c.NumTokens, idx.BM25Params)
		} else {
			freqs := countTokens(tokenize(c.Text))
			numTokens := 0
			for _, v := range freqs {
				numTokens += v
			}
			bm25Score = scoreChunkString(qtokens, freqs, numTokens, idx.BM25Params)
		}

		if bm25Score <= 0 {
			continue
		}

		// Aplicar field weight multiplier
		fw := fieldWeight(c, baseTokens)
		finalScore := bm25Score * fw

		pre = append(pre, scored{chunk: c, score: finalScore})
	}

	if len(pre) == 0 {
		return nil
	}

	sort.Slice(pre, func(i, j int) bool {
		return pre[i].score > pre[j].score
	})

	if len(pre) > topK {
		pre = pre[:topK]
	}

	results := make([]SearchResult, 0, len(pre))
	for _, s := range pre {
		snippet := extractSnippet(s.chunk.Text, baseTokens, 5)
		results = append(results, SearchResult{
			Path:    s.chunk.Path,
			Start:   s.chunk.Start,
			End:     s.chunk.End,
			Type:    s.chunk.Type,
			Name:    s.chunk.Name,
			Parent:  s.chunk.Parent,
			Imports: s.chunk.Imports,
			Symbols: s.chunk.Symbols,
			Lang:    s.chunk.Lang,
			Score:   s.score,
			Text:    s.chunk.Text,
			Snippet: snippet,
		})
	}
	return results
}
