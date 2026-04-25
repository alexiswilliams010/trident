// SPDX-License-Identifier: MIT
// This file lives under lib/ and MUST NOT be parsed during Phase 1
// (only the targeted dependency pass in Phase 3 should reach it).
pragma solidity ^0.8.20;

contract Test {
    function setUp() public virtual {}
}
