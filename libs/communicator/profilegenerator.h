#include "receiver.h"


class ProfileGenerator : public Receiver {
public:
    ProfileGenerator(const json &jsonConfigs);
    ~ProfileGenerator() override = default;

    void dispatchThread() override {
        std::thread handler(&ProfileGenerator::profileDataGenerator, this);
        handler.detach();
    }
    void profileDataGenerator();

    void loadConfigs(const json &jsonConfigs, bool isConstructing = true ) override;

private:
    uint16_t msvc_numWarmUpBatches, msvc_numProfileBatches;
    uint8_t msvc_inputRandomizeScheme;
    uint8_t msvc_stepMode = 0, msvc_step = 1;
};