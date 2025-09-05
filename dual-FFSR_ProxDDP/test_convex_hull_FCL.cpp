// Compute the minimum distance and its gradient between
// (chasersat, link1_3) and (chasersat, link1_4) using Pinocchio + FCL.
//
// - Loads MJCF from xml/dual_arm_space_robot_add_object.xml
// - Builds Pinocchio models and collision geometry
// - Adds collision pairs between the requested frames
// - Computes the minimum distance across groups and an analytic gradient
//   w.r.t. q (no finite differences), using witness points + frame Jacobians.

#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <fstream>

namespace pin = pinocchio;

static std::string find_xml_path()
{
  const char *fname = "dual_arm_space_robot_add_object.xml";
  std::vector<std::string> candidates;
  // Common run locations
  candidates.emplace_back(std::string("xml/") + fname);   // run from repo root
  candidates.emplace_back(std::string("../xml/") + fname); // run from build/

  // Path relative to this source file directory
  std::string file = __FILE__;
  auto pos = file.find_last_of("/\\");
  if(pos != std::string::npos) {
    std::string base = file.substr(0, pos);
    candidates.emplace_back(base + "/xml/" + fname);
  }

  for(const auto &p : candidates) {
    std::ifstream f(p.c_str());
    if(f.good()) return p;
  }
  throw std::runtime_error("Could not locate xml/" + std::string(fname) +
                           ". Tried xml/, ../xml/, and path relative to source file.");
}

static std::vector<int> geom_indices_for_frames(const pin::Model &model,
                                                const pin::GeometryModel &gmodel,
                                                const std::vector<std::string> &names)
{
  // 1) Exact frame names
  std::vector<pin::FrameIndex> wanted;
  wanted.reserve(names.size());
  for(const auto &n : names) {
    if(model.existFrame(n)) wanted.push_back(model.getFrameId(n));
  }

  std::vector<int> idxs;
  for(size_t i = 0; i < gmodel.geometryObjects.size(); ++i) {
    const auto &go = gmodel.geometryObjects[i];
    if(std::find(wanted.begin(), wanted.end(), go.parentFrame) != wanted.end())
      idxs.push_back(static_cast<int>(i));
  }
  if(!idxs.empty()) return idxs;

  // 2) Fallback: substring match
  std::vector<std::string> targets;
  targets.reserve(names.size());
  for(const auto &n : names) {
    std::string s = n; std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    targets.push_back(s);
  }
  for(size_t i = 0; i < gmodel.geometryObjects.size(); ++i) {
    const auto &go = gmodel.geometryObjects[i];
    std::string fname = model.frames[go.parentFrame].name;
    std::string gname = go.name;
    std::transform(fname.begin(), fname.end(), fname.begin(), ::tolower);
    std::transform(gname.begin(), gname.end(), gname.begin(), ::tolower);
    for(const auto &t : targets) {
      if(fname.find(t) != std::string::npos || gname.find(t) != std::string::npos) {
        idxs.push_back(static_cast<int>(i));
        break;
      }
    }
  }
  return idxs;
}

static std::vector<int> add_collision_pairs_between(pin::GeometryModel &gmodel,
                                                    const std::vector<int> &A,
                                                    const std::vector<int> &B)
{
  const int start = static_cast<int>(gmodel.collisionPairs.size());
  for(int ia : A) for(int ib : B) if(ia != ib) gmodel.addCollisionPair(pin::CollisionPair(ia, ib));
  std::vector<int> out;
  for(int k = start; k < (int)gmodel.collisionPairs.size(); ++k) out.push_back(k);
  return out;
}

struct PairResult {
  double dist{std::numeric_limits<double>::infinity()};
  Eigen::VectorXd grad; // size = nv
  int pair_index{-1};
};

static PairResult distance_and_gradient_for_pair(const pin::Model &model,
                                                 pin::Data &data,
                                                 const pin::GeometryModel &gmodel,
                                                 pin::GeometryData &gdata,
                                                 int cp_index,
                                                 const Eigen::VectorXd &q)
{
  // FK + placements
  pin::forwardKinematics(model, data, q);
  pin::updateFramePlacements(model, data);
  pin::updateGeometryPlacements(model, data, gmodel, gdata);

  // Make sure nearest points are available
  if(gdata.distanceRequests.size() == gmodel.collisionPairs.size()) {
    for(auto &req : gdata.distanceRequests) {
      req.enable_nearest_points = true; // newer FCL computes them anyway
      req.enable_signed_distance = false;
    }
  }

  // Compute distance for the given pair (fills gdata.distanceResults[cp_index])
  (void)pin::computeDistance(gmodel, gdata, cp_index);
  const auto &cp = gmodel.collisionPairs[cp_index];
  const auto &res = gdata.distanceResults[cp_index];

  const double dist = res.min_distance;

  // Witness points in local geom frames
  const Eigen::Vector3d p1_local = res.nearest_points[0];
  const Eigen::Vector3d p2_local = res.nearest_points[1];

  // World placements of the two geoms
  const pin::SE3 &oMg1 = gdata.oMg[cp.first];
  const pin::SE3 &oMg2 = gdata.oMg[cp.second];

  // World witness points
  const Eigen::Vector3d p1_world = oMg1.act(p1_local);
  const Eigen::Vector3d p2_world = oMg2.act(p2_local);

  // Prefer provided normal if non-zero, otherwise use (p2 - p1)/d
  Eigen::Vector3d n_world = Eigen::Vector3d::Zero();
  if(res.normal.norm() > 0.) n_world = res.normal.normalized();
  else if(std::abs(dist) > 0.) n_world = (p2_world - p1_world).normalized();

  // Parent frames of the geoms
  const int fid1 = gmodel.geometryObjects[cp.first].parentFrame;
  const int fid2 = gmodel.geometryObjects[cp.second].parentFrame;

  // Compute Jacobians at frames (LOCAL_WORLD_ALIGNED), then shift to witness points
  pin::computeJointJacobians(model, data);
  const auto RF = pin::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  Eigen::Matrix<double,6,Eigen::Dynamic> Jcol1 = pin::getFrameJacobian(model, data, fid1, RF);
  Eigen::Matrix<double,6,Eigen::Dynamic> Jcol2 = pin::getFrameJacobian(model, data, fid2, RF);

  // Offsets from frame origins to witness points (world)
  const pin::SE3 &oMf1 = data.oMf[fid1];
  const pin::SE3 &oMf2 = data.oMf[fid2];
  const Eigen::Vector3d r1 = p1_world - oMf1.translation();
  const Eigen::Vector3d r2 = p2_world - oMf2.translation();

  pin::SE3 jointToP1 = pin::SE3::Identity();
  jointToP1.translation() = r1;
  pin::SE3 jointToP2 = pin::SE3::Identity();
  jointToP2.translation() = r2;

  const Eigen::Matrix<double,6,6> Ainv1 = jointToP1.toActionMatrixInverse();
  const Eigen::Matrix<double,6,6> Ainv2 = jointToP2.toActionMatrixInverse();
  Eigen::Matrix<double,6,Eigen::Dynamic> Jpcol1 = Ainv1 * Jcol1;
  Eigen::Matrix<double,6,Eigen::Dynamic> Jpcol2 = Ainv2 * Jcol2;

  // Gradient of distance w.r.t q
  Eigen::RowVectorXd grad_row = n_world.transpose() * (Jpcol2.topRows<3>() - Jpcol1.topRows<3>());

  PairResult out;
  out.dist = dist;
  out.grad = grad_row.transpose();
  out.pair_index = cp_index;
  return out;
}

static PairResult best_over_pairs(const pin::Model &model,
                                  pin::Data &data,
                                  const pin::GeometryModel &gmodel,
                                  pin::GeometryData &gdata,
                                  const std::vector<int> &pairs,
                                  const Eigen::VectorXd &q)
{
  PairResult best;
  for(int cp : pairs) {
    PairResult pr = distance_and_gradient_for_pair(model, data, gmodel, gdata, cp, q);
    if(pr.dist < best.dist) best = pr;
  }
  return best;
}

int main()
{
  try {
    // Build models from MJCF
    pin::Model model; pin::GeometryModel gcoll, gvis;
    const std::string xml_path = find_xml_path();
    // Build from MJCF
    pin::mjcf::buildModel(xml_path, model);
    pin::mjcf::buildGeom(model, xml_path, pin::COLLISION, gcoll);
    pin::mjcf::buildGeom(model, xml_path, pin::VISUAL, gvis);

    // Zero gravity (optional)
    model.gravity.linear().setZero();

    // If collision geometry is empty, mirror from visual
    if(gcoll.geometryObjects.empty() && !gvis.geometryObjects.empty()) {
      for(const auto &go : gvis.geometryObjects) gcoll.addGeometryObject(go);
    }

    // Pre-compute local AABBs (if available through underlying FCL objects)
    for(auto &go : gcoll.geometryObjects) {
      if(go.geometry) {
        try { go.geometry->computeLocalAABB(); } catch(...) {}
      }
    }

    pin::Data data(model);

    // Lookup geometry indices for the requested frames
    const std::vector<std::string> chasers_names{"chasersat","chasers","base"};
    const std::vector<std::string> link13_names{"link1_3"};
    const std::vector<std::string> link14_names{"link1_4"};
    const auto chasers_idxs = geom_indices_for_frames(model, gcoll, chasers_names);
    const auto link13_idxs  = geom_indices_for_frames(model, gcoll, link13_names);
    const auto link14_idxs  = geom_indices_for_frames(model, gcoll, link14_names);

    if(chasers_idxs.empty()) throw std::runtime_error("No geoms for 'chasersat'.");
    if(link13_idxs.empty())  throw std::runtime_error("No geoms for 'link1_3'.");
    if(link14_idxs.empty())  throw std::runtime_error("No geoms for 'link1_4'.");

    // Add collision pairs for both groups
    const auto pairs13 = add_collision_pairs_between(gcoll, chasers_idxs, link13_idxs);
    const auto pairs14 = add_collision_pairs_between(gcoll, chasers_idxs, link14_idxs);

    // Create geometry data AFTER adding pairs so result/request sizes match
    pin::GeometryData gdata(gcoll);

    // Neutral configuration
    const Eigen::VectorXd q0 = pin::neutral(model);

    // Compute best pair per group (min distance + gradient)
    const PairResult best13 = best_over_pairs(model, data, gcoll, gdata, pairs13, q0);
    const PairResult best14 = best_over_pairs(model, data, gcoll, gdata, pairs14, q0);

    const bool pick13 = (best13.dist <= best14.dist);
    const double d1 = best13.dist;
    const double d2 = best14.dist;
    const double dmin = pick13 ? best13.dist : best14.dist;
    const Eigen::VectorXd grad = pick13 ? best13.grad : best14.grad;
    const std::string which = pick13 ? "(chasersat, link1_3)" : "(chasersat, link1_4)";

    std::cout << "Distance chasersat-link1_3: " << d1 << '\n';
    std::cout << "Distance chasersat-link1_4: " << d2 << '\n';
    std::cout << "System min distance: " << dmin << " from " << which << '\n';
    std::cout << "Gradient (ddmin/dq) size: " << grad.size() << '\n';
    std::cout << "Gradient: [";
    for(int i=0;i<grad.size();++i){ std::cout << grad[i]; if(i+1<grad.size()) std::cout << ", "; }
    std::cout << "]\n";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
