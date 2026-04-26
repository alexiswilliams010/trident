package main

import (
	"fmt"

	"github.com/example/myapp/internal/utils"
	"github.com/pkg/errors"
)

type Calculator struct {
	Base int
}

// Method on Calculator — qualified_name should be "<file>.Calculator.Add"
// thanks to the receiver-prefix logic.
func (c *Calculator) Add(n int) int {
	return c.Base + utils.Double(n)
}

// Cross-file call edge: DoubleIt → utils.Double should be tier-A `certain`
// after Phase 3.
func (c *Calculator) DoubleIt() int {
	return utils.Double(c.Base)
}

// Greet uses utils.Helper to exercise selector_expression resolution.
func (c *Calculator) Greet(name string) string {
	return utils.Helper(name)
}

func main() {
	c := &Calculator{Base: utils.X}
	fmt.Println(c.DoubleIt())
	fmt.Println(c.Greet("world"))
	if err := errors.New("boom"); err != nil {
		fmt.Println(err)
	}
}
