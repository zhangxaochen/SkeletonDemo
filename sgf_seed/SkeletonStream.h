#ifndef SKELETONSTREAM_H
#define SKELETONSTREAM_H

#include <SenseKit/Plugins/PluginKit.h>
#include <SenseKit/SenseKit.h>
#include <SenseKitUL/skul_ctypes.h>
#include <SenseKitUL/streams/skeleton_types.h>

namespace sensekit { namespace plugins { namespace skeleton {

    class SkeletonStream : public sensekit::plugins::SingleBinStream<sensekit_skeletonframe_wrapper_t,
                                                                     sensekit_skeleton_joint_t>
    {
    public:
        SkeletonStream(sensekit::PluginServiceProxy& pluginService,
                       sensekit::Sensor streamSet,
                       sensekit_stream_t sourceStream,
                       size_t skeletonCount)
            : SingleBinStream(pluginService,
                              streamSet,
                              sensekit::StreamDescription(SENSEKIT_STREAM_SKELETON,
                                                          DEFAULT_SUBTYPE),
                              sizeof(sensekit_skeleton_t) * skeletonCount)

        {}
    };
}}}

#endif /* SKELETONSTREAM_H */
