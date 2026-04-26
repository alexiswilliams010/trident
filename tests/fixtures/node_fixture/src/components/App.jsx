// JSX in a JS file (.jsx) — tests cross-language relative resolution
// (.jsx imports a .tsx sibling).
import { Button } from "./Button";

export function App() {
  return <Button label="hi" onClick={() => undefined} />;
}
