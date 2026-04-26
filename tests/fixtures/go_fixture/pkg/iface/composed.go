package iface

// ReadWriter embeds two interfaces declared in the same file (basic.go).
// Cross-file: when basic.go and composed.go live in the same package,
// the embedded names resolve via Phase 3 cross-file inheritance.
type ReadWriter interface {
	Reader
	Writer
}

// FullIO embeds another interface declared in this same file (intra-file
// resolution) and one declared in basic.go (cross-file).
//
// It also redeclares Read(...) — same name as Reader.Read — exercising the
// override-edge generation path between an embedded interface's method and
// the embedder's directly-declared method of the same name.
type FullIO interface {
	ReadWriter
	Closer
	Read(p []byte) (int, error)
}
