// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Mix of import styles for Phase 3 classification:
//   - relative   → intra_repo
//   - "@..."     → external by prefix
//   - "lib/..."  → external by dep-dir prefix
//   - bare      → external best-effort (Phase 7 Foundry remappings get the precise answer)
import {Token} from "./Token.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "lib/forge-std/src/Test.sol";
import "forge-std/Test.sol";

contract External {
    Token public t;
}
