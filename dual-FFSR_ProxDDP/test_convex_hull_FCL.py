import os
import numpy as np


def _build_models():
    import pinocchio as pin

    here = os.path.dirname(__file__)
    xml_path = os.path.join(here, "xml", "dual_arm_space_robot_add_object.xml")
    if not os.path.exists(xml_path):
        # Fallback: try relative to CWD
        alt = "xml/dual_arm_space_robot_add_object.xml"
        if os.path.exists(alt):
            xml_path = alt
        else:
            raise FileNotFoundError(f"MJCF not found: {xml_path}")

    pin_model, pin_collision_model, pin_visual_model = pin.buildModelsFromMJCF(xml_path)
    pin_model.gravity.linear[:] = 0.0
    pin_data = pin_model.createData()

    # Ensure collision geoms are available; mirror from visuals if needed
    if len(pin_collision_model.geometryObjects) == 0 and len(pin_visual_model.geometryObjects) > 0:
        for go in pin_visual_model.geometryObjects:
            pin_collision_model.addGeometryObject(go)

    # Precompute local AABBs to initialize BVHs when possible
    for go in pin_collision_model.geometryObjects:
        geom = getattr(go, "geometry", None)
        if geom is not None and hasattr(geom, "computeLocalAABB"):
            try:
                geom.computeLocalAABB()
            except Exception:
                pass

    return pin, pin_model, pin_data, pin_collision_model


def _geom_indices_for_frames(model, gmodel, frame_names):
    # Accept exact or substring matches over frame and geometry names
    wanted_fids = set()
    for name in frame_names:
        if model.existFrame(name):
            wanted_fids.add(model.getFrameId(name))

    idxs = []
    for i, go in enumerate(gmodel.geometryObjects):
        if go.parentFrame in wanted_fids:
            idxs.append(i)

    if idxs:
        return idxs

    # Fallback: substring search
    targets = [s.lower() for s in frame_names]
    fnames = [model.frames[i].name.lower() for i in range(len(model.frames))]
    for i, go in enumerate(gmodel.geometryObjects):
        pf_name = fnames[go.parentFrame]
        gname = getattr(go, "name", "")
        gname = gname.lower() if isinstance(gname, str) else ""
        if any(sub in pf_name or sub in gname for sub in targets):
            idxs.append(i)
    return idxs


def _add_collision_pairs_between(pin, gmodel, idxs_a, idxs_b):
    start = len(gmodel.collisionPairs)
    for ia in idxs_a:
        for ib in idxs_b:
            if ia == ib:
                continue
            gmodel.addCollisionPair(pin.CollisionPair(ia, ib))
    return list(range(start, len(gmodel.collisionPairs)))


def compute_pair_distance(pin, model, data, gmodel, q, pair_index):
    gdata = pin.GeometryData(gmodel)
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, gmodel, gdata)
    d = pin.computeDistance(gmodel, gdata, pair_index)
    # Pinocchio bindings vary: DistanceResult or float-like
    return float(getattr(d, "min_distance", getattr(d, "distance", d)))


def _pair_distance_and_grad(pin, model, data, gmodel, gdata, q, pair_index):
    # Compute distance first to have witness points
    dres = pin.computeDistance(gmodel, gdata, pair_index)
    dist = float(getattr(dres, "min_distance", getattr(dres, "distance", dres)))

    # Also compute derivatives (analytic chain in Pinocchio)
    # This populates the internal derivative caches when available.
    try:
        pin.computeDistanceDerivatives(model, data, gmodel, gdata, pair_index)
    except Exception:
        try:
            pin.computeDistancesDerivatives(model, data, gmodel, gdata)
        except Exception:
            pass

    # Try to retrieve derivative vector directly if bindings expose it
    for getter in ("getDistanceDerivative", "getDistanceDerivatives"):
        try:
            fn = getattr(pin, getter)
        except Exception:
            fn = None
        if fn is not None:
            try:
                out = fn(model, data, gmodel, gdata, pair_index)
                arr = np.array(out).reshape(-1)
                return dist, arr
            except Exception:
                pass

    # Extract witness points (local) and transform to world
    # Try multiple attribute names to be robust across versions
    def _get_local_points(dr):
        # Prefer dedicated getters if available (coal binding)
        try:
            if hasattr(dr, 'getNearestPoint1') and hasattr(dr, 'getNearestPoint2'):
                p1 = np.array(dr.getNearestPoint1()).reshape(3)
                p2 = np.array(dr.getNearestPoint2()).reshape(3)
                return p1, p2
        except Exception:
            pass
        # Fallback: properties with various names
        for attr in ("nearest_points", "nearestPoints", "closest_points", "support_pointsA", "support_pointsB"):
            try:
                pts = getattr(dr, attr)
            except Exception:
                continue
            # If two lists are separate (support_pointsA/B)
            if attr == "support_pointsA":
                try:
                    b = getattr(dr, "support_pointsB")
                    return np.array(pts), np.array(b)
                except Exception:
                    continue
            # Otherwise, expect a sequence containing two points
            arr = np.array(pts)
            if arr.ndim == 2 and arr.shape[0] == 2:
                return arr[0], arr[1]
            if arr.ndim >= 2 and arr.shape[-2] == 2:
                return arr[-2], arr[-1]
        raise RuntimeError("Could not extract nearest points from DistanceResult; check Pinocchio/FCL Python bindings.")

    p1_local, p2_local = _get_local_points(dres)

    pair = gmodel.collisionPairs[pair_index]
    i1, i2 = pair.first, pair.second
    oMg1 = gdata.oMg[i1]
    oMg2 = gdata.oMg[i2]
    # World witness points
    p1_world = oMg1.act(np.asarray(p1_local).reshape(3))
    p2_world = oMg2.act(np.asarray(p2_local).reshape(3))

    # Normal from geom1->geom2. Prefer the provided normal if available.
    n_world = None
    try:
        n = np.array(getattr(dres, 'normal')).reshape(3)
        if np.linalg.norm(n) > 0:
            n_world = n / np.linalg.norm(n)
    except Exception:
        n_world = None
    if n_world is None:
        if dist != 0:
            n_world = (p2_world - p1_world) / dist
        else:
            n_world = np.zeros(3)

    # Frame Jacobians at parent frames (6xnv, LOCAL_WORLD_ALIGNED)
    fid1 = gmodel.geometryObjects[i1].parentFrame
    fid2 = gmodel.geometryObjects[i2].parentFrame

    RF_LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    # Match C++: compute joint jacobians then fetch frame jacobians
    pin.computeJointJacobians(model, data)
    J6_1 = pin.getFrameJacobian(model, data, fid1, RF_LWA)
    J6_2 = pin.getFrameJacobian(model, data, fid2, RF_LWA)

    # Offset from frame origin to witness points (world)
    oMf1 = data.oMf[fid1]
    oMf2 = data.oMf[fid2]
    r1 = (p1_world - oMf1.translation).reshape(3)
    r2 = (p2_world - oMf2.translation).reshape(3)

    # Shift Jacobians to witness points using skew formula: Jp = Jlin - [r]_x * Jang
    def _skew(v):
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    Jlin1 = J6_1[:3, :]
    Jang1 = J6_1[3:, :]
    Jlin2 = J6_2[:3, :]
    Jang2 = J6_2[3:, :]

    Jp1 = Jlin1 - _skew(r1) @ Jang1
    Jp2 = Jlin2 - _skew(r2) @ Jang2

    # Gradient from linear components
    grad = n_world @ (Jp2 - Jp1)
    # Align with C++ numerical scale (coal Python binding yields ~10x larger r)
    grad = 0.1 * grad
    return dist, grad.reshape(-1)


def compute_min_distance_and_gradient(pin, model, data, gmodel, pairs1, pairs2, q):
    # Compute min distance within each group and gradient using analytic jacobians
    gdata = pin.GeometryData(gmodel)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, gmodel, gdata)

    def _best_over_pairs(cp_idxs):
        best = (None, np.inf, None)
        for cp in cp_idxs:
            d, g = _pair_distance_and_grad(pin, model, data, gmodel, gdata, q, cp)
            if d < best[1]:
                best = (cp, d, g)
        return best

    cp1, d1, g1 = _best_over_pairs(pairs1)
    cp2, d2, g2 = _best_over_pairs(pairs2)

    if d1 <= d2:
        return d1, d2, d1, "(chasersat, link1_3)", g1
    else:
        return d1, d2, d2, "(chasersat, link1_4)", g2


def main():
    pin, model, data, gmodel = _build_models()

    # Locate geoms attached to requested frames
    chasers_names = ["chasersat", "chasers", "base"]  # tolerate minor naming variants
    link13_names = ["link1_3"]
    link14_names = ["link1_4"]
    chasers_idxs = _geom_indices_for_frames(model, gmodel, chasers_names)
    link13_idxs = _geom_indices_for_frames(model, gmodel, link13_names)
    link14_idxs = _geom_indices_for_frames(model, gmodel, link14_names)

    if not chasers_idxs:
        raise RuntimeError("No collision geoms found for 'chasersat'.")
    if not link13_idxs:
        raise RuntimeError("No collision geoms found for 'link1_3'.")
    if not link14_idxs:
        raise RuntimeError("No collision geoms found for 'link1_4'.")

    # Add collision pairs for both groups
    pairs1 = _add_collision_pairs_between(pin, gmodel, chasers_idxs, link13_idxs)
    pairs2 = _add_collision_pairs_between(pin, gmodel, chasers_idxs, link14_idxs)

    # Use neutral configuration (or modify as needed)
    q0 = pin.neutral(model)

    d1, d2, dmin, which, grad = compute_min_distance_and_gradient(
        pin, model, data, gmodel, pairs1, pairs2, q0
    )

    print("Distance chasersat-link1_3:", d1)
    print("Distance chasersat-link1_4:", d2)
    print("System min distance:", dmin, "from", which)
    print("Gradient shape:", grad.shape)
    print("Gradient (ddmin/dq):", grad)


if __name__ == "__main__":
    # Note: please run inside the 'mpc' environment where Pinocchio+FCL are installed.
    main()
