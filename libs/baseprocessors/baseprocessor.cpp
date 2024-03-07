#include "baseprocessor.h"

/**
 * @brief Construct a new Base Preprocessor that inherites the LocalGPUDataMicroservice given the `InType`
 * 
 * @param configs 
 */
BaseProcessor::BaseProcessor(const BaseMicroserviceConfigs &configs) : Microservice(configs) {}

/**
 * @brief Check if the request is still worth being processed.
 * For instance, if the request is already late at the moment of checking, there is no value in processing it anymore.
 * 
 * @tparam InType 
 * @return true 
 * @return false 
 */
bool BaseProcessor::checkReqEligibility(ClockType currReq_gentime) {
    return true;
}