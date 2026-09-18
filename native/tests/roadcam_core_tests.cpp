#include "roadcam_core.h"

#include <cassert>
#include <limits>
#include <cstring>

int main() {
    RoadCamBoundingBox valid_box{100, 200, 300, 200};
    RoadCamBoundingBox invalid_box{-1, 200, 300, 200};
    assert(roadcam_bounding_box_is_valid(&valid_box) == 1);
    assert(roadcam_bounding_box_is_valid(&invalid_box) == 0);
    assert(roadcam_bounding_box_is_valid(nullptr) == 0);

    char plate[ROADCAM_CORE_PLATE_CAPACITY]{};
    roadcam_normalize_plate(" VA ABC-1234 ", plate, ROADCAM_CORE_PLATE_CAPACITY);
    assert(std::strcmp(plate, "VAABC1234") == 0);
    roadcam_normalize_plate("abc", plate, 3);
    assert(std::strcmp(plate, "AB") == 0);

    RoadCamHandle *handle = roadcam_create(0.90F);
    assert(handle != nullptr);
    assert(roadcam_create(1.01F) == nullptr);
    RoadCamVehicleDetection detection{ROADCAM_VEHICLE_CAR, 0.94F, valid_box, 7U, 1.25};
    RoadCamKnownVehicle known[] = {{42, "VA ABC-1234"}};
    RoadCamResult result = roadcam_process_detection(handle, &detection, "va.abc1234", known, 1U);
    assert(result.accepted == 1);
    assert(result.known_vehicle_id == 42);
    assert(std::strcmp(result.normalized_plate, "VAABC1234") == 0);

    detection.confidence = 0.89F;
    result = roadcam_process_detection(handle, &detection, nullptr, nullptr, 0U);
    assert(result.accepted == 0);
    detection.confidence = 0.94F;
    detection.bounding_box.width = 0;
    result = roadcam_process_detection(handle, &detection, nullptr, nullptr, 0U);
    assert(result.accepted == 0);
    detection.bounding_box.width = 300;
    detection.confidence = std::numeric_limits<float>::quiet_NaN();
    result = roadcam_process_detection(handle, &detection, nullptr, nullptr, 0U);
    assert(result.accepted == 0);
    roadcam_destroy(handle);
    return 0;
}
