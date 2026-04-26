// JSX in a TS file (.tsx). Verifies the .tsx sub-grammar dispatch and that
// JSX usage of `Button` in App.jsx will eventually link cross-file.
import * as React from "react";

interface ButtonProps {
  label: string;
  onClick: () => void;
}

export function Button(props: ButtonProps): JSX.Element {
  return <button onClick={props.onClick}>{props.label}</button>;
}
