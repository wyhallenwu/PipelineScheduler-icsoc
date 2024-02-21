#include "baseprocessor.h"

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @tparam InType 
 * @param configs 
 */
template<typename InType>
BaseProcessor<InType>::BaseProcessor(const BaseMicroserviceConfigs &configs) : LocalGPUDataMicroservice<InType>(configs) {}

/**
 * @brief Check if the request is still worth being processed.
 * For instance, if the request is already late at the moment of checking, there is no value in processing it anymore.
 * 
 * @tparam InType 
 * @return true 
 * @return false 
 */
template<typename InType>
bool BaseProcessor<InType>::checkReqEligibility(ClockTypeTemp currReq_gentime) {
    return true;
}