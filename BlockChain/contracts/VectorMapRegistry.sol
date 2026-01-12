// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title VectorMapRegistry
/// @notice Maintains a mapping from a logical vector map path (OSS storage path)
///         to a manifest CID on IPFS that binds original vector files and
///         copyright artifacts (zeroWatermark.png, copyright.png, copyright.txt).
contract VectorMapRegistry {
    // Use keccak256(path) as key to avoid dynamic string keys in mapping
    mapping(bytes32 => string) private _pathHashToCid;

    event VectorMapRegistered(
        bytes32 indexed pathHash,
        string path,
        string cid,
        address indexed operator,
        uint256 timestamp
    );

    /// @dev Compute the mapping key for a given path
    function _keyFor(string memory path) internal pure returns (bytes32) {
        return keccak256(bytes(path));
    }

    /// @notice Set or update the manifest CID for a given logical path
    /// @param path Logical OSS path, e.g. "bucket/Institution/Railway"
    /// @param cid Manifest CID stored on IPFS
    function registerVectorMap(string calldata path, string calldata cid) external {
        require(bytes(path).length > 0, "path empty");
        require(bytes(cid).length > 0, "cid empty");
        bytes32 key = _keyFor(path);
        _pathHashToCid[key] = cid;
        emit VectorMapRegistered(key, path, cid, msg.sender, block.timestamp);
    }

    /// @notice Get the manifest CID for a given logical path
    /// @param path Logical OSS path
    /// @return cid Manifest CID or empty string if not set
    function getManifestCID(string calldata path) external view returns (string memory cid) {
        cid = _pathHashToCid[_keyFor(path)];
    }
}


