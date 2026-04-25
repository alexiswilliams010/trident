// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IPolicy {
    function isPolicyActive() external view returns (bool);
}

abstract contract Policy is IPolicy {
    address public manager;

    modifier onlyManager() {
        require(msg.sender == manager, "not manager");
        _;
    }

    function isPolicyActive() public view virtual override returns (bool) {
        return manager != address(0);
    }

    function _requireSender(address sender) internal view {
        require(sender == manager, "bad sender");
    }
}

contract SingleExecutorPolicy is Policy {
    address public executor;

    function isPolicyActive() public view override returns (bool) {
        return super.isPolicyActive() && executor != address(0);
    }

    function setExecutor(address _executor) external onlyManager {
        executor = _executor;
    }
}
