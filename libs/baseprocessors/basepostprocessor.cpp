#include "basepostprocessor.h"

template<typename InType>
BasePostprocessor<InType>::BasePostprocessor(const BaseMicroserviceConfigs &configs) : Microservice<InType>(configs) {
}