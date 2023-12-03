/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#pragma once

#include <fstream>

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wmissing-braces"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
#pragma GCC diagnostic pop

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif
//frk
#define CL_TARGET_BATCH_TIME 0.3F // seconds

namespace dev
{
namespace eth
{
class CLMiner : public Miner
{
public:

    CLMiner(unsigned _index, CLSettings _settings, DeviceDescriptor& _device);
    ~CLMiner() override;

    static void enumDevices_old(std::map<string, DeviceDescriptor>& _DevicesCollection);
    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);

protected:
    bool initDevice() override;

    bool initEpoch_internal() override;

    void kick_miner() override;
    void kick_miner_frk();

private:
    
    void workLoop() override;
    void workLoop_old();
    bool initEpoch_frk();
    bool initDevice_frk();

    vector<cl::Context> m_context;
    vector<cl::CommandQueue> m_queue;
    vector<cl::CommandQueue> m_abortqueue;
    cl::Kernel m_searchKernel;
    cl::Kernel m_dagKernel;
    cl::Device m_device;

    vector<cl::Buffer> m_dag;
    vector<cl::Buffer> m_light;
    vector<cl::Buffer> m_header;
    vector<cl::Buffer> m_searchBuffer;
//frk
    void free_buffers() {
        m_abortMutex.lock();

        m_header.clear();
        m_searchBuffer.clear();
        m_queue.clear();
        m_context.clear();
        m_abortqueue.clear();
    }
    std::mutex m_abortMutex;
//    
    void clear_buffer() {
        m_dag.clear();
        m_light.clear();
        m_header.clear();
        m_searchBuffer.clear();
        m_queue.clear();
        m_context.clear();
        m_abortqueue.clear();
    }

    CLSettings m_settings;

    unsigned m_dagItems = 0;
    uint64_t m_lastNonce = 0;

};

}  // namespace eth
}  // namespace dev
