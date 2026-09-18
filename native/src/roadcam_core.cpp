#include "roadcam_core.h"

#include <cmath>
#include <cstring>
#include <memory>

struct RoadCamHandle {
    explicit RoadCamHandle(float threshold) : minimum_confidence(threshold) {}
    float minimum_confidence;
};

namespace {

bool is_supported_vehicle_type(RoadCamVehicleType type) {
    return type == ROADCAM_VEHICLE_CAR || type == ROADCAM_VEHICLE_TRUCK ||
           type == ROADCAM_VEHICLE_BUS || type == ROADCAM_VEHICLE_MOTORCYCLE;
}

RoadCamResult empty_result() {
    RoadCamResult result{};
    result.accepted = 0;
    result.known_vehicle_id = 0;
    result.normalized_plate[0] = '\0';
    return result;
}

}  // namespace

extern "C" RoadCamHandle *roadcam_create(float minimum_confidence) {
    if (minimum_confidence < 0.0F || minimum_confidence > 1.0F) {
        return nullptr;
    }
    return new RoadCamHandle(minimum_confidence);
}

extern "C" void roadcam_destroy(RoadCamHandle *handle) {
    delete handle;
}

extern "C" int roadcam_bounding_box_is_valid(const RoadCamBoundingBox *bounding_box) {
    return bounding_box != nullptr && bounding_box->x >= 0 && bounding_box->y >= 0 &&
           bounding_box->width > 0 && bounding_box->height > 0;
}

extern "C" void roadcam_normalize_plate(const char *input, char *output, unsigned int output_capacity) {
    if (output == nullptr || output_capacity == 0U) {
        return;
    }
    unsigned int index = 0;
    if (input != nullptr) {
        for (const unsigned char *current = reinterpret_cast<const unsigned char *>(input);
             *current != '\0' && index + 1U < output_capacity; ++current) {
            if (*current >= 'a' && *current <= 'z') {
                output[index++] = static_cast<char>(*current - ('a' - 'A'));
            } else if ((*current >= 'A' && *current <= 'Z') ||
                       (*current >= '0' && *current <= '9')) {
                output[index++] = static_cast<char>(*current);
            }
        }
    }
    output[index] = '\0';
}

extern "C" RoadCamResult roadcam_process_detection(
    const RoadCamHandle *handle,
    const RoadCamVehicleDetection *detection,
    const char *raw_plate,
    const RoadCamKnownVehicle *known_vehicles,
    unsigned int known_vehicle_count) {
    RoadCamResult result = empty_result();
    if (handle == nullptr || detection == nullptr ||
        !is_supported_vehicle_type(detection->vehicle_type) ||
        !std::isfinite(detection->confidence) ||
        detection->confidence < handle->minimum_confidence ||
        detection->confidence > 1.0F ||
        roadcam_bounding_box_is_valid(&detection->bounding_box) == 0) {
        return result;
    }

    result.accepted = 1;
    roadcam_normalize_plate(raw_plate, result.normalized_plate, ROADCAM_CORE_PLATE_CAPACITY);
    if (result.normalized_plate[0] == '\0' || known_vehicles == nullptr) {
        return result;
    }
    for (unsigned int index = 0; index < known_vehicle_count; ++index) {
        char normalized_known_plate[ROADCAM_CORE_PLATE_CAPACITY]{};
        roadcam_normalize_plate(known_vehicles[index].plate, normalized_known_plate,
                                ROADCAM_CORE_PLATE_CAPACITY);
        if (std::strcmp(result.normalized_plate, normalized_known_plate) == 0) {
            result.known_vehicle_id = known_vehicles[index].id;
            break;
        }
    }
    return result;
}
