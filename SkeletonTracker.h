#ifndef SKELETONTRACKER_H
#define SKELETONTRACKER_H

#include <SenseKit/Plugins/PluginKit.h>
#include <SenseKit/SenseKit.h>
#include <SenseKitUL/skul_ctypes.h>
#include <SenseKitUL/streams/Depth.h>
#include <SenseKitUL/streams/skeleton_types.h>
#include "SkeletonStream.h"

namespace sensekit { namespace plugins { namespace skeleton {

    class SkeletonTracker : public sensekit::FrameReadyListener
    {
    public:
        static const size_t MAX_SKELETONS;

        SkeletonTracker(PluginServiceProxy& pluginService,
                        Sensor streamSet,
                        sensekit_stream_t sourceStream)
            : m_pluginService(pluginService),
              m_reader(streamSet.create_reader()),
              m_sourceStreamHandle(sourceStream),
              m_sensor(streamSet)
        {
            m_depthStream = m_reader.stream<sensekit::DepthStream>();
            m_depthStream.start();

            m_reader.addListener(*this);
            m_skeletonStream = std::make_unique<SkeletonStream>(m_pluginService,
                                                                m_sensor,
                                                                m_sourceStreamHandle,
                                                                SkeletonTracker::MAX_SKELETONS);
        }

        sensekit_stream_t sourceStream() { return m_sourceStreamHandle; }

        virtual void on_frame_ready(sensekit::StreamReader& reader, sensekit::Frame& frame) override;

    private:
        sensekit_stream_t m_sourceStreamHandle;
        DepthStream m_depthStream{nullptr};
        StreamReader m_reader;
        PluginServiceProxy& m_pluginService;
        Sensor m_sensor;

        using SkeletonStreamPtr = std::unique_ptr<SkeletonStream>;
        SkeletonStreamPtr m_skeletonStream;

		//zhangxaochen:
		//CoordinateMapper m_mapper;
    };


}}}


#endif /* SKELETONTRACKER_H */
