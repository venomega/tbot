package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

var pyChunkerSrc = `
import ast, json, sys

def chunk_py_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        source = f.read()
    lines = source.split("\n")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks = []
    module_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.add(alias.name)
            else:
                module_imports.add(node.module or "")

    # Module docstring / imports chunk
    first_def = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def = node
            break

    if first_def and first_def.lineno > 1:
        chunk = {
            "start": 1,
            "end": first_def.lineno - 1,
            "type": "module",
            "name": "",
            "parent": "",
            "symbols": [],
            "text": "\n".join(lines[:first_def.lineno - 1]),
        }
        chunks.append(chunk)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = _node_end(node, lines)
            symbols = [node.name]
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        symbols.append(n.func.id)
            chunks.append({
                "start": node.lineno,
                "end": end,
                "type": "function",
                "name": node.name,
                "parent": "",
                "symbols": symbols,
                "text": "\n".join(lines[node.lineno - 1:end]),
            })
        elif isinstance(node, ast.ClassDef):
            class_end = _node_end(node, lines)
            methods = [n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and hasattr(n, "lineno")]
            chunks.append({
                "start": node.lineno,
                "end": class_end,
                "type": "class",
                "name": node.name,
                "parent": "",
                "symbols": methods,
                "text": "\n".join(lines[node.lineno - 1:class_end]),
            })

    for c in chunks:
        c["imports"] = sorted(module_imports) if module_imports else []
    return chunks

def _node_end(node, lines):
    max_line = len(lines)
    end = getattr(node, "end_lineno", None)
    if end:
        return min(end, max_line)
    end = node.lineno + 1
    for n in ast.walk(node):
        e = getattr(n, "end_lineno", None) or getattr(n, "lineno", 0)
        if e > end:
            end = e
    return min(end, max_line)

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            path = req.get("path", "")
            if path:
                chks = chunk_py_file(path)
                print(json.dumps({"chunks": chks, "path": path}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e), "path": req.get("path", "")}), flush=True)
`

var pyChunkerPath string

func init() {
	home, err := os.UserHomeDir()
	if err == nil {
		pyChunkerPath = filepath.Join(home, ".config", "tbot", "rag_index", "_rag_chunker.py")
	}
}

type pythonChunk struct {
	Start   int      `json:"start"`
	End     int      `json:"end"`
	Type    string   `json:"type"`
	Name    string   `json:"name"`
	Parent  string   `json:"parent"`
	Symbols []string `json:"symbols"`
	Imports []string `json:"imports"`
	Text    string   `json:"text"`
}

type pythonResponse struct {
	Path   string        `json:"path"`
	Chunks []pythonChunk `json:"chunks"`
	Error  string        `json:"error,omitempty"`
}

type pythonRequest struct {
	Path string `json:"path"`
}

var pyProcess *exec.Cmd
var pyStdin *bufio.Writer
var pyScanner *bufio.Scanner

func startPythonChunker() error {
	if pyProcess != nil {
		return nil
	}
	if pyChunkerPath == "" {
		return fmt.Errorf("cannot determine home directory")
	}
	if err := os.MkdirAll(filepath.Dir(pyChunkerPath), 0755); err != nil {
		return err
	}
	if err := os.WriteFile(pyChunkerPath, []byte(pyChunkerSrc), 0644); err != nil {
		return err
	}
	cmd := exec.Command("python3", pyChunkerPath)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	pyProcess = cmd
	pyStdin = bufio.NewWriter(stdin)
	pyScanner = bufio.NewScanner(stdout)
	pyScanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	return nil
}

func stopPythonChunker() {
	if pyProcess != nil {
		pyStdin.Flush()
		pyProcess.Process.Kill()
		pyProcess.Wait()
		pyProcess = nil
		pyStdin = nil
		pyScanner = nil
	}
}

func chunkPython(path string) ([]*Chunk, error) {
	if err := startPythonChunker(); err != nil {
		return chunkFallback(path)
	}
	req := pythonRequest{Path: path}
	data, _ := json.Marshal(req)
	pyStdin.Write(data)
	pyStdin.WriteByte('\n')
	pyStdin.Flush()

	if !pyScanner.Scan() {
		return chunkFallback(path)
	}
	var resp pythonResponse
	if err := json.Unmarshal(pyScanner.Bytes(), &resp); err != nil {
		return chunkFallback(path)
	}
	if resp.Error != "" || resp.Chunks == nil {
		return chunkFallback(path)
	}
	if resp.Path != path {
		return chunkFallback(path)
	}
	chunks := make([]*Chunk, len(resp.Chunks))
	for i, pc := range resp.Chunks {
		chunks[i] = &Chunk{
			Path:    path,
			Start:   pc.Start,
			End:     pc.End,
			Type:    pc.Type,
			Name:    pc.Name,
			Parent:  pc.Parent,
			Symbols: pc.Symbols,
			Imports: pc.Imports,
			Lang:    "python",
			Text:    pc.Text,
		}
	}
	return chunks, nil
}

func chunkGoFile(path string) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	imports := make([]string, 0, len(f.Imports))
	for _, imp := range f.Imports {
		if imp.Path != nil {
			imports = append(imports, strings.Trim(imp.Path.Value, "\""))
		}
	}

	var chunks []*Chunk

	// Module header (package + imports + doc)
	if f.Doc != nil || len(imports) > 0 {
		firstDecl := len(src)
		if len(f.Decls) > 0 {
			firstDecl = fset.Position(f.Decls[0].Pos()).Offset
		}
		text := string(src[:firstDecl])
		text = strings.TrimSpace(text)
		if text != "" {
			startLine := 1
			endLine := startLine + strings.Count(text, "\n")
			chunks = append(chunks, &Chunk{
				Path:    path,
				Start:   startLine,
				End:     endLine,
				Type:    "module",
				Name:    f.Name.Name,
				Parent:  "",
				Imports: imports,
				Symbols: []string{},
				Lang:    "go",
				Text:    text,
			})
		}
	}

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				continue
			}
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					start := fset.Position(s.Pos()).Line
					end := fset.Position(s.End()).Line
					typeName := s.Name.Name
					var symbols []string
					var chunkType string
					if st, ok := s.Type.(*ast.StructType); ok {
						chunkType = "struct"
						for _, field := range st.Fields.List {
							if len(field.Names) > 0 {
								symbols = append(symbols, field.Names[0].Name)
							}
						}
					} else if _, ok := s.Type.(*ast.InterfaceType); ok {
						chunkType = "interface"
					} else {
						chunkType = "type"
					}
					text := getText(src, fset, s.Pos(), s.End())
					chunks = append(chunks, &Chunk{
						Path:    path,
						Start:   start,
						End:     end,
						Type:    chunkType,
						Name:    typeName,
						Parent:  "",
						Imports: imports,
						Symbols: symbols,
						Lang:    "go",
						Text:    text,
					})

				case *ast.ValueSpec:
					if len(s.Names) == 0 {
						continue
					}
					start := fset.Position(s.Pos()).Line
					end := fset.Position(s.End()).Line
					name := s.Names[0].Name
					text := getText(src, fset, s.Pos(), s.End())
					chunks = append(chunks, &Chunk{
						Path:    path,
						Start:   start,
						End:     end,
						Type:    "var",
						Name:    name,
						Parent:  "",
						Imports: imports,
						Symbols: []string{name},
						Lang:    "go",
						Text:    text,
					})
				}
			}

		case *ast.FuncDecl:
			start := fset.Position(d.Pos()).Line
			end := fset.Position(d.End()).Line
			chunkType := "function"
			parent := ""
			if d.Recv != nil && len(d.Recv.List) > 0 {
				chunkType = "method"
				recvType := exprString(d.Recv.List[0].Type)
				parent = recvType
			}
			text := getText(src, fset, d.Pos(), d.End())
			symbols := []string{d.Name.Name}

			chunks = append(chunks, &Chunk{
				Path:    path,
				Start:   start,
				End:     end,
				Type:    chunkType,
				Name:    d.Name.Name,
				Parent:  parent,
				Imports: imports,
				Symbols: symbols,
				Lang:    "go",
				Text:    text,
			})
		}
	}

	return chunks, nil
}

var (
	jsFuncRegex      = regexp.MustCompile(`(?:async\s+)?function\s*\*?\s*(\w+)\s*\(`)
	jsClassRegex     = regexp.MustCompile(`class\s+(\w+)`)
	jsMethodRegex    = regexp.MustCompile(`(\w+)\s*\([^)]*\)\s*\{`)
	rustFnRegex      = regexp.MustCompile(`(?:pub\s+)?(?:unsafe\s+)?fn\s+(\w+)`)
	rustStructRegex  = regexp.MustCompile(`(?:pub\s+)?struct\s+(\w+)`)
	rustImplRegex    = regexp.MustCompile(`(?:pub\s+)?impl(?:\s*<[^>]*>)?\s+(\w+)`)
	rustTraitRegex   = regexp.MustCompile(`(?:pub\s+)?trait\s+(\w+)`)
	mdHeadingRegex   = regexp.MustCompile(`^(#{1,6})\s+(.+)$`)
	chapterRegex     = regexp.MustCompile(`(?i)^(chapter|capítulo|lección|lesson|unit|unidad|parte)\s+\d+`)
	allCapsLine      = regexp.MustCompile(`^[A-ZÁÉÍÓÚÜÑ\s]{4,}$`)
	separatorRegex   = regexp.MustCompile(`^(---|\*\*\*|___)\s*$`)
	boldTitleRegex   = regexp.MustCompile(`^(\*{1,3}|_{1,3})(.+)(\*{1,3}|_{1,3})\s*$`)
)

// default chunk size/overlap for documents (can be overridden via CLI)
var defaultChunkSize = 100
var defaultOverlap = 50

func chunkJSFile(path string) ([]*Chunk, error) {
	return chunkBraceFile(path, "javascript", jsFuncRegex, jsClassRegex)
}

func chunkTSFile(path string) ([]*Chunk, error) {
	return chunkBraceFile(path, "typescript", jsFuncRegex, jsClassRegex)
}

func chunkRustFile(path string) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(src), "\n")

	var chunks []*Chunk
	type rbBlock struct {
		name      string
		start     int
		blockType string
		depth     int
	}
	var stack []rbBlock
	depth := 0
	inImpl := false

	rustDefRe := regexp.MustCompile(`(?:pub\s+)?(?:unsafe\s+)?(fn|struct|impl|trait|enum|mod|type)\s+(?:<[^>]*>\s+)?(\w+)`)

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		m := rustDefRe.FindStringSubmatch(trimmed)
		if m != nil {
			blockType := m[1]
			name := m[2]
			if blockType == "fn" && inImpl {
				blockType = "method"
			}
			stack = append(stack, rbBlock{name: name, start: i + 1, blockType: blockType, depth: depth})
			if blockType == "impl" {
				inImpl = true
			}
		}

		for _, r := range line {
			if r == '{' {
				depth++
			} else if r == '}' {
				depth--
				if depth < 0 {
					depth = 0
				}
			}
		}

		for len(stack) > 0 && depth <= stack[len(stack)-1].depth {
			bb := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if i+1-bb.start >= 3 {
				text := strings.Join(lines[bb.start-1:i], "\n")
				chunks = append(chunks, &Chunk{
					Path:  path,
					Start: bb.start,
					End:   i + 1,
					Type:  bb.blockType,
					Name:  bb.name,
					Lang:  "rust",
					Text:  text,
				})
			}
			if bb.blockType == "impl" {
				inImpl = false
			}
		}
	}
	if len(chunks) == 0 {
		return chunkFallback(path)
	}
	return chunks, nil
}

func chunkBraceFile(path, lang string, funcRe, classRe *regexp.Regexp) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(src), "\n")
	var chunks []*Chunk

	type braceBlock struct {
		name      string
		start     int
		blockType string
		depth     int
	}

	var stack []braceBlock
	depth := 0
	inClass := false

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		funcMatch := funcRe.FindStringSubmatch(trimmed)
		classMatch := classRe.FindStringSubmatch(trimmed)

		var blockType string
		var name string
		if classMatch != nil {
			blockType = "class"
			name = classMatch[1]
		} else if funcMatch != nil {
			blockType = "function"
			name = funcMatch[1]
		}

		if blockType != "" {
			stack = append(stack, braceBlock{name: name, start: i + 1, blockType: blockType, depth: depth})
		}

		for _, r := range line {
			if r == '{' {
				depth++
			} else if r == '}' {
				depth--
				if depth < 0 {
					depth = 0
				}
			}
		}

		for len(stack) > 0 && depth <= stack[len(stack)-1].depth {
			bb := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if i+1-bb.start >= 3 {
				chunkType := bb.blockType
				parent := ""
				if chunkType == "function" && inClass {
					chunkType = "method"
					parent = findClassInRange(lines, bb.start)
				}
				text := strings.Join(lines[bb.start-1:i], "\n")
				chunks = append(chunks, &Chunk{
					Path:    path,
					Start:   bb.start,
					End:     i + 1,
					Type:    chunkType,
					Name:    bb.name,
					Parent:  parent,
					Symbols: []string{},
					Lang:    lang,
					Text:    text,
				})
			}
		}

		if classMatch != nil {
			inClass = true
		}
	}

	if len(chunks) == 0 {
		return chunkLinear(path, 30, 15)
	}
	return chunks, nil
}

func findClassInRange(lines []string, lineNum int) string {
	for i := lineNum - 1; i >= 0 && i >= lineNum-10; i-- {
		m := jsClassRegex.FindStringSubmatch(strings.TrimSpace(lines[i]))
		if m != nil {
			return m[1]
		}
	}
	return ""
}

func extractRustImports(lines []string) []string {
	re := regexp.MustCompile(`^(?:pub\s+)?(?:use|extern\s+crate)\s+(.+?)(?:\s+as\s+\w+)?;`)
	var imports []string
	for _, line := range lines {
		m := re.FindStringSubmatch(strings.TrimSpace(line))
		if m != nil {
			imports = append(imports, m[1])
		}
	}
	return imports
}

// isSectionHeader checks if a line looks like a section heading.
// Returns the section name if it is a header, empty string otherwise.
func isSectionHeader(line string) (bool, string) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return false, ""
	}
	// 1. Standard markdown headings: # Title
	if m := mdHeadingRegex.FindStringSubmatch(line); m != nil {
		return true, m[2]
	}
	// 2. Chapter/part patterns: "Parte 1", "Capítulo 2", "Lección 3", etc.
	if chapterRegex.MatchString(trimmed) {
		return true, trimmed
	}
	// 3. ALL CAPS lines (longer than 10 chars)
	if len(trimmed) > 10 && allCapsLine.MatchString(trimmed) {
		return true, trimmed
	}
	// 4. Bold/italic title lines: **Title**, *Title*, __Title__
	if m := boldTitleRegex.FindStringSubmatch(trimmed); m != nil {
		inner := strings.TrimSpace(m[2])
		if len(inner) > 3 {
			return true, inner
		}
	}
	return false, ""
}

func chunkMD(path string) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(src), "\n")
	var chunks []*Chunk
	var start int = 1
	var headingName string

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Check for separator lines: ---, ***, ___
		if separatorRegex.MatchString(trimmed) {
			if start < i {
				text := strings.Join(lines[start-1:i-1], "\n")
				text = strings.TrimSpace(text)
				if text != "" {
					chunks = append(chunks, &Chunk{
						Path:  path,
						Start: start,
						End:   i,
						Type:  "section",
						Name:  headingName,
						Lang:  "markdown",
						Text:  text,
					})
				}
			}
			start = i + 1
			headingName = "" // separator clears the heading name
			continue
		}

		// Check for section headers (markdown headings, chapter patterns, ALL CAPS, bold titles)
		if isHeader, name := isSectionHeader(line); isHeader {
			if start < i {
				text := strings.Join(lines[start-1:i-1], "\n")
				text = strings.TrimSpace(text)
				if text != "" {
					chunks = append(chunks, &Chunk{
						Path:  path,
						Start: start,
						End:   i,
						Type:  "section",
						Name:  headingName,
						Lang:  "markdown",
						Text:  text,
					})
				}
			}
			start = i + 1
			headingName = name
		}
	}

	// Last chunk (remaining content after last header)
	if start <= len(lines) {
		text := strings.Join(lines[start-1:], "\n")
		text = strings.TrimSpace(text)
		if text != "" {
			chunks = append(chunks, &Chunk{
				Path:  path,
				Start: start,
				End:   len(lines),
				Type:  "section",
				Name:  headingName,
				Lang:  "markdown",
				Text:  text,
			})
		}
	}

	// Fallback: if no sections found, use paragraph-based splitting
	if len(chunks) == 0 {
		return chunkParagraph(path, defaultChunkSize)
	}

	// Apply overlap between adjacent chunks if configured
	if defaultOverlap > 0 {
		chunks = applyOverlap(chunks, lines, defaultOverlap)
	}

	return chunks, nil
}

// chunkParagraph splits a document by double newlines (paragraphs),
// merging small paragraphs up to minSize lines.
func chunkParagraph(path string, minSize int) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	content := string(src)
	paragraphs := strings.Split(content, "\n\n")
	var chunks []*Chunk
	line := 1
	var pending *Chunk

	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			line++ // account for blank line
			continue
		}
		numLines := strings.Count(p, "\n") + 1

		if pending != nil {
			// Try merging small paragraphs
			pendingLines := pending.End - pending.Start + 1
			if pendingLines+numLines <= minSize*2 {
				// Merge into pending chunk
				pending.Text = strings.TrimSpace(pending.Text + "\n\n" + p)
				pending.End = line + numLines - 1
				line += numLines + 1
				continue
			}
			// Pending is big enough, flush it
			chunks = append(chunks, pending)
			pending = nil
		}

		// Skip very small paragraphs (likely formatting artifacts)
		if numLines < 3 && len(p) < 80 {
			line += numLines + 1
			continue
		}

		pending = &Chunk{
			Path:  path,
			Start: line,
			End:   line + numLines - 1,
			Type:  "paragraph",
			Lang:  "markdown",
			Text:  p,
		}
		line += numLines + 1
	}

	if pending != nil {
		chunks = append(chunks, pending)
	}

	if len(chunks) == 0 {
		return chunkLinear(path, defaultChunkSize, defaultOverlap)
	}
	return chunks, nil
}

// applyOverlap adds lines from before each chunk as context.
// The chunk's Start/End metadata remains the same, but the Text
// includes up to `overlap` extra lines from before the chunk start.
func applyOverlap(chunks []*Chunk, lines []string, overlap int) []*Chunk {
	if len(chunks) <= 1 || overlap <= 0 {
		return chunks
	}
	result := make([]*Chunk, len(chunks))
	for i, c := range chunks {
		// Extend text backward by up to `overlap` lines
		textStart := c.Start - overlap
		if textStart < 1 {
			textStart = 1
		}
		// Don't go earlier than the previous chunk's start (to avoid huge bloat)
		if i > 0 && textStart < chunks[i-1].Start {
			textStart = chunks[i-1].Start
		}
		text := strings.Join(lines[textStart-1:c.End], "\n")
		result[i] = &Chunk{
			Path:    c.Path,
			Start:   c.Start,
			End:     c.End,
			Type:    c.Type,
			Name:    c.Name,
			Parent:  c.Parent,
			Imports: c.Imports,
			Symbols: c.Symbols,
			Lang:    c.Lang,
			Text:    text,
		}
	}
	return result
}

func chunkTxt(path string) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(src), "\n")
	var chunks []*Chunk

	inChapter := false
	chapterStart := 1
	chapterName := ""

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if chapterRegex.MatchString(trimmed) || (len(trimmed) > 10 && allCapsLine.MatchString(trimmed)) {
			if inChapter && i+1-chapterStart > 5 {
				text := strings.Join(lines[chapterStart-1:i], "\n")
				chunks = append(chunks, &Chunk{
					Path:  path,
					Start: chapterStart,
					End:   i,
					Type:  "chapter",
					Name:  chapterName,
					Lang:  "text",
					Text:  text,
				})
			}
			chapterStart = i + 1
			chapterName = trimmed
			inChapter = true
		}
	}
	if inChapter && len(lines)+1-chapterStart > 5 {
		text := strings.Join(lines[chapterStart-1:], "\n")
		chunks = append(chunks, &Chunk{
			Path:  path,
			Start: chapterStart,
			End:   len(lines),
			Type:  "chapter",
			Name:  chapterName,
			Lang:  "text",
			Text:  text,
		})
	}
	if len(chunks) == 0 {
		// Split by double newlines
		paragraphs := strings.Split(string(src), "\n\n")
		line := 1
		for _, p := range paragraphs {
			p = strings.TrimSpace(p)
			if len(p) < 50 {
				line += strings.Count(p, "\n") + 2
				continue
			}
			numLines := strings.Count(p, "\n") + 1
			chunks = append(chunks, &Chunk{
				Path:  path,
				Start: line,
				End:   line + numLines - 1,
				Type:  "paragraph",
				Lang:  "text",
				Text:  p,
			})
			line += numLines + 2
		}
	}
	if len(chunks) == 0 {
		return chunkLinear(path, defaultChunkSize, defaultOverlap)
	}
	return chunks, nil
}

func chunkLinear(path string, chunkSize, overlap int) ([]*Chunk, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(src), "\n")
	if len(lines) == 0 {
		return nil, nil
	}
	var chunks []*Chunk
	start := 1
	for start <= len(lines) {
		end := start + chunkSize - 1
		if end > len(lines) {
			end = len(lines)
		}
		text := strings.Join(lines[start-1:end], "\n")
		chunks = append(chunks, &Chunk{
			Path:  path,
			Start: start,
			End:   end,
			Type:  "code_block",
			Lang:  detectLang(path),
			Text:  text,
		})
		if end == len(lines) {
			break
		}
		start = end - overlap
		if start < 1 {
			start = 1
		}
	}
	return chunks, nil
}

func chunkFallback(path string) ([]*Chunk, error) {
	return chunkLinear(path, 30, 15)
}

var langMap = map[string]string{
	".go": "go", ".py": "python", ".js": "javascript", ".ts": "typescript",
	".jsx": "jsx", ".tsx": "tsx", ".rs": "rust", ".java": "java",
	".rb": "ruby", ".php": "php", ".c": "c", ".h": "c", ".cpp": "cpp",
	".hpp": "cpp", ".cs": "csharp", ".swift": "swift", ".kt": "kotlin",
	".scala": "scala", ".r": "r", ".lua": "lua", ".sh": "bash",
	".bash": "bash", ".zsh": "bash", ".fish": "bash", ".ps1": "powershell",
	".pl": "perl", ".pm": "perl", ".tex": "latex", ".md": "markdown",
	".rst": "rst", ".txt": "text", ".json": "json", ".yaml": "yaml",
	".yml": "yaml", ".xml": "xml", ".html": "html", ".htm": "html",
	".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
	".sql": "sql", ".toml": "toml", ".ini": "ini", ".cfg": "ini",
	".env": "env", ".dockerfile": "dockerfile",
}

func detectLang(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	if l, ok := langMap[ext]; ok {
		return l
	}
	base := strings.ToLower(filepath.Base(path))
	if base == "dockerfile" || strings.HasPrefix(base, "dockerfile") {
		return "dockerfile"
	}
	if strings.HasPrefix(base, "makefile") || base == "gnumakefile" {
		return "makefile"
	}
	return "unknown"
}

func chunkFile(path string) ([]*Chunk, error) {
	ext := strings.ToLower(filepath.Ext(path))
	base := strings.ToLower(filepath.Base(path))

	// check file size
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.Size() > 500*1024 {
		return nil, nil
	}

	switch ext {
	case ".go":
		return chunkGoFile(path)
	case ".py":
		return chunkPython(path)
	case ".js", ".jsx", ".mjs", ".cjs":
		return chunkJSFile(path)
	case ".ts", ".tsx", ".mts", ".cts":
		return chunkTSFile(path)
	case ".rs":
		return chunkRustFile(path)
	case ".md", ".mdx":
		return chunkMD(path)
	case ".txt":
		return chunkTxt(path)
	default:
		if ext == "" {
			if base == "dockerfile" || strings.HasPrefix(base, "dockerfile") {
				return chunkFallback(path)
			}
			if strings.HasPrefix(base, "makefile") || base == "gnumakefile" {
				return chunkFallback(path)
			}
		}
		return chunkFallback(path)
	}
}

func getText(src []byte, fset *token.FileSet, pos, end token.Pos) string {
	startOff := fset.Position(pos).Offset
	endOff := fset.Position(end).Offset
	if startOff < 0 || endOff > len(src) || startOff >= endOff {
		return ""
	}
	return string(src[startOff:endOff])
}

type identVisitor struct {
	names []string
}

func (v *identVisitor) Visit(n ast.Node) ast.Visitor {
	if id, ok := n.(*ast.Ident); ok {
		v.names = append(v.names, id.Name)
	}
	return v
}

func exprString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.StarExpr:
		return "*" + exprString(e.X)
	case *ast.IndexExpr:
		return exprString(e.X) + "[" + exprString(e.Index) + "]"
	case *ast.SelectorExpr:
		return exprString(e.X) + "." + e.Sel.Name
	default:
		return fmt.Sprintf("%T", e)
	}
}
