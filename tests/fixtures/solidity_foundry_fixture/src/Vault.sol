// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Token} from "./Token.sol";

contract Vault {
    Token public immutable token;
    mapping(address => uint256) public balances;

    event Deposit(address indexed user, uint256 amount);
    event Withdraw(address indexed user, uint256 amount);

    constructor(Token _token) {
        token = _token;
    }

    function deposit(uint256 amount) external {
        token.transfer(address(this), amount);
        balances[msg.sender] += amount;
        emit Deposit(msg.sender, amount);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        token.transfer(msg.sender, amount);
        emit Withdraw(msg.sender, amount);
    }
}
