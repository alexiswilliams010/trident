package utils

func GreetTwice(name string) string {
	return Helper(name) + " / " + Helper(name)
}
