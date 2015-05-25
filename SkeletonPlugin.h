#ifndef SKELETONPLUGIN_H
#define SKELETONPLUGIN_H

#include <SenseKit/Plugins/PluginKit.h>
#include <SenseKitUL/SenseKitUL.h>
#include "SkeletonTracker.h"
#include <memory>
#include <vector>

namespace sensekit { namespace plugins { namespace skeleton {

    class SkeletonPlugin : public sensekit::PluginBase
    {
    public:
        static const size_t MAX_SKELETONS = 5;

        SkeletonPlugin(PluginServiceProxy* pluginProxy)
            : PluginBase(pluginProxy)
        {
            register_for_stream_events();
        }

        virtual ~SkeletonPlugin()
        {
            unregister_for_stream_events();
        }

    private:
        virtual void on_stream_added(sensekit_streamset_t setHandle,
                                     sensekit_stream_t streamHandle,
                                     sensekit_stream_desc_t desc) override;

        virtual void on_stream_removed(sensekit_streamset_t setHandle,
                                       sensekit_stream_t streamHandle,
                                       sensekit_stream_desc_t desc) override;

        using SkeletonTrackerPtr = std::unique_ptr<SkeletonTracker>;
        using SkeletonTrackerList = std::vector<SkeletonTrackerPtr>;

        SkeletonTrackerList m_skeletonTrackers;
    };
}}}


#endif /* SKELETONPLUGIN_H */
