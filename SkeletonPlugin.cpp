#include "SkeletonPlugin.h"
#include <SenseKit/SenseKit.h>

EXPORT_PLUGIN(sensekit::plugins::skeleton::SkeletonPlugin);

namespace sensekit { namespace plugins { namespace skeleton {

    void SkeletonPlugin::on_stream_added(sensekit_streamset_t setHandle,
                                         sensekit_stream_t streamHandle,
                                         sensekit_stream_desc_t desc)
    {
        if (desc.type != SENSEKIT_STREAM_DEPTH)
            return; // if new stream is not depth, we don't care.

        m_skeletonTrackers.push_back(std::make_unique<SkeletonTracker>(get_pluginService(),
                                                                     Sensor(setHandle),
                                                                     streamHandle));
    }

    void SkeletonPlugin::on_stream_removed(sensekit_streamset_t setHandle,
                                           sensekit_stream_t streamHandle,
                                           sensekit_stream_desc_t desc)
    {
        auto it = std::find_if(m_skeletonTrackers.cbegin(),
                               m_skeletonTrackers.cend(),
                               [&streamHandle] (const SkeletonTrackerPtr& trackerPtr)
                               {
                                   return trackerPtr->sourceStream() == streamHandle;
                               });

        if (it != m_skeletonTrackers.cend())
        {
            m_skeletonTrackers.erase(it);
        }
    }

}}}
