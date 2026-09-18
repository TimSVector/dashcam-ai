#ifndef ROADCAM_CORE_H
#define ROADCAM_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#define ROADCAM_CORE_PLATE_CAPACITY 64

typedef struct RoadCamHandle RoadCamHandle;

typedef enum RoadCamVehicleType {
    ROADCAM_VEHICLE_UNKNOWN = 0,
    ROADCAM_VEHICLE_CAR = 1,
    ROADCAM_VEHICLE_TRUCK = 2,
    ROADCAM_VEHICLE_BUS = 3,
    ROADCAM_VEHICLE_MOTORCYCLE = 4
} RoadCamVehicleType;

typedef struct RoadCamBoundingBox {
    int x;
    int y;
    int width;
    int height;
} RoadCamBoundingBox;

typedef struct RoadCamVehicleDetection {
    RoadCamVehicleType vehicle_type;
    float confidence;
    RoadCamBoundingBox bounding_box;
    unsigned long long frame_number;
    double timestamp_seconds;
} RoadCamVehicleDetection;

typedef struct RoadCamKnownVehicle {
    int id;
    const char *plate;
} RoadCamKnownVehicle;

typedef struct RoadCamResult {
    int accepted;
    int known_vehicle_id;
    char normalized_plate[ROADCAM_CORE_PLATE_CAPACITY];
} RoadCamResult;

/* Returns NULL when minimum_confidence is outside [0, 1]. */
RoadCamHandle *roadcam_create(float minimum_confidence);
void roadcam_destroy(RoadCamHandle *handle);

/* Returns non-zero if x/y are non-negative and width/height are positive. */
int roadcam_bounding_box_is_valid(const RoadCamBoundingBox *bounding_box);

/* ASCII uppercase, retaining only A-Z and 0-9. Always NUL terminates output. */
void roadcam_normalize_plate(const char *input, char *output, unsigned int output_capacity);

/*
 * Makes a deterministic decision. A detection is accepted only when its box is
 * valid, its vehicle type is supported, and confidence meets the handle's
 * threshold. When raw_plate is supplied, it is normalized and exactly matched
 * against normalized known-vehicle plates.
 */
RoadCamResult roadcam_process_detection(
    const RoadCamHandle *handle,
    const RoadCamVehicleDetection *detection,
    const char *raw_plate,
    const RoadCamKnownVehicle *known_vehicles,
    unsigned int known_vehicle_count);

#ifdef __cplusplus
}
#endif

#endif
