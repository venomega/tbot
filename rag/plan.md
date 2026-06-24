## Análisis Completo del RAG Actual

### Lo que ya funciona bien ✅
- BM25 sólido: con inverted index, packed frequencies y dos paths de scoring.
- Chunkers por lenguaje: AST en Go, AST vía Python para .py, brace-matching para JS/TS/Rust.
- Stemming español básico: sufijos comunes para reducir variantes morfológicas.
- Snippets contextuales: extrae líneas alrededor del mejor match.
- Indexación paralela: usa runtime.NumCPU() workers.
- Formato compacto: packed frequencies (uint32) y term IDs numéricos.

───

## Áreas de Mejora

### 1. 🏗️ Chunking (la base de todo)

Problema: Solo chunkMD aplica overlap. Los chunkers de código (Go, Python, JS, Rust) no añaden contexto del padre. Si buscas un método, no ves de qué clase es.

Solución: Añadir overlap contextual a todos los chunkers de código. Por ejemplo, para un método en Go, incluir las primeras líneas del struct padre.

```go
// En chunkGoFile, al crear un chunk de método:
if parent != "" {
    // Incluir header del struct/func antes del método como contexto
    c.Text = parentContext + "\n" + c.Text
}
```

Problema: Chunks de 100 líneas (default) pueden ser muy grandes para docs, muy pequeños para código denso.

Solución: Chunk size adaptativo según tipo de archivo y densidad de tokens.

───

### 2. 🔍 Retrieval Híbrido (BM25 + Embeddings)

Problema: BM25 es puramente lexical. "Crear usuario" y "registrar nuevo usuario" no matchean aunque sean semánticamente iguales.

Solución: Añadir dense retrieval como segunda etapa o como fusión híbrida:

```
┌─────────┐     ┌──────────┐     ┌──────────┐
│  BM25   │────▶│ Re-ranking│────▶│ Resultados│
│ (rápido)│     │ (opcional)│     │ finales  │
└─────────┘     └──────────┘     └──────────┘
     │
     ▼
┌─────────┐
│Dense (op)│ (usando sentence-transformers vía subprocess como el chunker Python)
└─────────┘
```

Se puede implementar un hybrid score: score = α  BM25 + (1-α)  cosine_sim(embedding)

O más simple: re-ranking con embeddings solo sobre los top-50 de BM25.

───

### 3. 🧠 Tokenización y Stemming

Problemas específicos:

┌────────────────────────┬─────────────────────────────────────────────────────┬────────────────────────────┐
│ Problema               │ Ejemplo                                             │ Impacto                    │
├────────────────────────┼─────────────────────────────────────────────────────┼────────────────────────────┤
│ Acentos sin normalizar │ "configuración" ≠ "configuracion"                   │ ❌ No matchean             │
│ Sin stop words         │ "el", "la", "de", "que" ocupan espacio en el índice │ Inflación de términos      │
│ Stemming muy básico    │ "producción" ≠ "producir" (no comparten raíz)       │ Falsos negativos           │
│ CamelCase sin dividir  │ "getUserData" es un solo token                      │ No aparece buscando "user" │
└────────────────────────┴─────────────────────────────────────────────────────┴────────────────────────────┘

Soluciones:

```go
// Normalizar acentos ANTES de tokenizar
var accentMap = map[rune]rune{
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    'ü': 'u', 'ñ': 'n',
}

func normalize(text string) string {
    return strings.Map(func(r rune) rune {
        if n, ok := accentMap[unicode.ToLower(r)]; ok {
            return n
        }
        return r
    }, text)
}

// Dividir camelCase ANTES de stem
func splitCamelCase(token string) []string {
    // "getUserData" → ["get", "User", "Data"]
    // luego se stemiza cada parte
}
```

Mejora drástica: Usar Snowball Spanish stemmer (implementación en C puro, se puede linkar con cgo o llamar vía subprocess). El actual es una lista de ≈40 sufijos; Snowball tiene reglas morfológicas completas.

───

### 4. 📊 Indexado Incremental

Problema: Cada vez que se indexa se re-indexa TODO. Con 29MB de índice, para proyectos grandes puede ser lento.

Solución: Usar fileMTimes (ya se guardan) para detectar cambios:

```go
func indexDirIncremental(root string) (*Index, error) {
    existing, err := loadIndex()
    if err != nil {
        return indexDir(root) // full rebuild
    }
    
    // Solo re-indexar archivos modificados o nuevos
    for _, path := range changedFiles(root, existing.Files) {
        chunks, err := chunkFile(path)
        // actualizar chunks en el índice existente
    }
    
    // Recalcular BM25 params solo si hubo cambios
    return idx, nil
}
```

───

### 5. 🌐 Expansión de Consulta (Query Expansion)

Idea: Antes de buscar, expandir la query con sinónimos relevantes para español:

```go
var spanishSynonyms = map[string][]string{
    "crear":   {"crear", "generar", "construir", "hacer", "elaborar"},
    "buscar":  {"buscar", "encontrar", "localizar", "consultar"},
    "archivo": {"archivo", "fichero", "documento"},
}

func expandQuery(query string) []string {
    tokens := tokenize(query)
    expanded := make(map[string]bool)
    for _, t := range tokens {
        expanded[t] = true
        if syns, ok := spanishSynonyms[t]; ok {
            for _, s := range syns {
                expanded[s] = true
            }
        }
    }
    // convertir a slice
}
```

Esto aumenta recall significativamente sin cambiar el índice.

───

### 6. 🎯 Pesado por Campos (Field-Weighted BM25)

Problema: Un match en Symbols (nombre de función) o Name debería valer más que en el cuerpo del código.

Solución: Multiplicar el score de BM25 según dónde ocurre el match:

```go
func fieldWeight(chunk *Chunk, queryTokens []string) float64 {
    weight := 1.0
    for _, t := range queryTokens {
        if strings.Contains(strings.ToLower(chunk.Name), t) {
            weight += 2.0  // Nombre de función pesa 3x
        }
        for _, s := range chunk.Symbols {
            if strings.Contains(strings.ToLower(s), t) {
                weight += 1.5  // Símbolo pesa 2.5x
            }
        }
    }
    return weight
}
```

───

### 7. 🔄 Re-ranking con Cross-Encoder

Si se quiere mejorar precisión sin cambiar el índice, añadir un segundo paso sobre los top-K de BM25 usando un cross-encoder ligero. Se puede llamar a Python (como ya se hace para el chunker) con un modelo pequeño tipo cross-encoder/ms-marco-MiniLM-L-6-v2.

No es necesario para búsqueda normal, pero para respuestas críticas mejora mucho.

───

### 8. 📐 Snippets Inteligentes

Problema: El snippet actual busca la línea con más query tokens y muestra ±5 líneas. Si la línea es una llave }, no es informativo.

Mejora: Usar el mejor párrafo semántico o función completa como snippet:

```go
// En lugar de extractSnippet actual:
func extractSnippet(text string, queryTokens []string, contextLines int) string {
    // 1. Encontrar la función/método/estructura completa que contiene el mejor match
    // 2. Si no se encuentra, usar el bloque delimitado por líneas en blanco
    // 3. Fallback al sistema actual
}
```

───

### 9. 🐛 Robustez del Python Chunker

Problema: El chunker Python se comunica por stdin/stdout. Si Python crashea o tarda, no hay timeout ni recovery.

Mejora:
- Añadir timeout por archivo en Go (context with deadline)
- Si Python falla N veces seguidas, hacer fallback permanente a chunkFallback
- Usar un pool de procesos Python (ya hay uno, pero sin límite de requests)

───

### 10. 🧪 Tests y Benchmarking

Problema: No hay tests.

Mínimo necesario:
```go
func TestTokenizeSpanish(t *testing.T) {
    cases := []struct{
        input, expected string
    }{
        {"configuración", "configura"},  // acentos + stemming
        {"creando", "cre"},              // gerundio
        {"archivos", "archiv"},          // plural
    }
    // ...
}

func TestSearchRelevancia(t *testing.T) {
    // Indexar un corpus pequeño conocido
    // Verificar que "buscar usuario" rankea más alto el chunk
    // que habla de usuarios que el que habla de configuración
}
```

───

### 11. ⚡ Optimizaciones de Performance

┌─────────────────────────────────────────────────────┬────────────────────────────────┬──────────────┐
│ Mejora                                              │ Impacto                        │ Dificultad   │
├─────────────────────────────────────────────────────┼────────────────────────────────┼──────────────┤
│ Skip list en inverted index (ordenado por chunk ID) │ Acelera conjunción de términos │ Media        │
│ Índice particionado por tipo de archivo             │ Búsqueda en subconjuntos       │ Baja         │
│ MMap para el índice                                 │ Menor RAM                      │ Alta         │
│ Concurrencia en search                              │ Búsquedas paralelas            │ Media        │
└─────────────────────────────────────────────────────┴────────────────────────────────┴──────────────┘

───

### 12. 🗄️ Soporte para Más Formatos

El RAG actual soporta Go, Python, JS/TS, Rust, MD, TXT. Podría añadirse:
- PDF (usando pdftotext o la skill de PDF)
- EPUB/RTF (convertir a texto)
- CSV/JSON (extraer celdas como texto plano)
- YAML/TOML (parsear como estructuras)

───

## Resumen de Prioridades Recomendadas

┌─────────────┬────────────────────────────────┬────────────┬───────────────────┐
│ Prioridad   │ Mejora                         │ Esfuerzo   │ Impacto           │
├─────────────┼────────────────────────────────┼────────────┼───────────────────┤
│ 🔴 Alta     │ Normalización de acentos       │ 1 hora     │ Muy alto          │
│ 🔴 Alta     │ División camelCase             │ 1 hora     │ Muy alto (código) │
│ 🔴 Alta     │ Stop words español             │ 30 min     │ Alto              │
│ 🟡 Media    │ Overlap en chunkers de código  │ 2 horas    │ Alto              │
│ 🟡 Media    │ Stemming Snowball (o mejorado) │ 4 horas    │ Muy alto          │
│ 🟡 Media    │ Query expansion con sinónimos  │ 2 horas    │ Alto              │
│ 🟢 Baja     │ Field-weighted scoring         │ 3 horas    │ Medio             │
│ 🟢 Baja     │ Indexado incremental           │ 6 horas    │ Medio             │
│ 🟢 Baja     │ Dense retrieval híbrido        │ 8-16 horas │ Alto              │
│ 🟢 Baja     │ Tests y benchmarks             │ 4 horas    │ Medio             │
└─────────────┴────────────────────────────────┴────────────┴───────────────────┘

───
