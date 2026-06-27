(function (global) {
  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function safeVec3(x, y, z) {
    return new THREE.Vector3(x || 0, y || 0, z || 0);
  }

  function arrToVec(a) {
    if (!a) return null;
    return new THREE.Vector3(
      (a[0] - 0.5) * 2.0,
      -(a[1] - 0.5) * 2.0,
      -(a[2] || 0) * 2.0
    );
  }

  function getPoseVec(frame, idx) {
    if (!frame || !frame.pose || !frame.pose[String(idx)]) return null;
    return arrToVec(frame.pose[String(idx)]);
  }

  function getHandVec(frame, side, idx) {
    const src = side === "left" ? frame.left_hand : frame.right_hand;
    if (!src || !src[idx]) return null;
    return arrToVec(src[idx]);
  }

  function resolveBone(bones, names) {
    for (const n of names) {
      if (bones[n]) return bones[n];
    }
    return null;
  }

  function slerpBoneToDir(bone, targetDir, amount) {
    if (!bone || !targetDir || targetDir.lengthSq() < 1e-8) return;
    const from = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion().setFromUnitVectors(from, targetDir.clone().normalize());