#include <trtengine.h>
#include "../baseprocessors/basepreprocessor.h"

template<typename InType>
class YoloV5Preprocessor : public BasePreprocessor<InType> {
public:
    YoloV5Preprocessor(const BaseMicroserviceConfigs &config);
    ~YoloV5Preprocessor();
protected:
    void batchRequests();
    // 
    // bool isTimeToBatch() override;
    //
    // bool checkReqEligibility(uint64_t currReq_genTime) override;
    //
    // void updateReqRate(ClockTypeTemp lastInterReqDuration) override;
};