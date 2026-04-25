// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Policy} from "./Policy.sol";

// Cross-file inheritance: extends `Policy` from another source unit.
contract LoggingPolicy is Policy {
    event Checked(address sender);

    function isPolicyActive() public view override returns (bool) {
        return manager != address(0);
    }

    function check(address sender) external view {
        _requireSender(sender);
    }
}
