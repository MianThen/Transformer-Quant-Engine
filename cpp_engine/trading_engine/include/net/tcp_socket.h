#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "transport/io_types.h"

namespace te {

// 非阻塞 TCP socket 封装。
//
// 跨平台事件多路复用:Linux 用 epoll,Windows 用 IOCP/WSAPoll。
// 本骨架用 Reactor 风格抽象,平台细节藏在 .cpp 里(条件编译)。
//
// 用途:Feed Handler 接收行情、Order Gateway 收发订单与回报。
class TcpSocket {
public:
    TcpSocket() = default;
    ~TcpSocket();

    // 禁止拷贝(持有 OS 句柄),允许移动
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;
    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;

    // 作为客户端连接到 host:port。成功返回 true。
    bool connect(const std::string& host, uint16_t port);

    // 设置非阻塞模式
    bool set_nonblocking(bool on);

    // 关闭 Nagle(TCP_NODELAY),低延迟场景必开
    bool set_no_delay(bool on);
    bool set_keepalive(bool on);
    bool set_buffer_sizes(int receive_bytes, int send_bytes);

    // 非阻塞读写:返回实际字节数,-1 表示错误,0 且 would_block 表示暂无数据
    ssize_t recv(void* buf, size_t len);
    ssize_t send(const void* buf, size_t len);
    ssize_t recvv(const MutableIoSlice* slices, size_t count);
    ssize_t sendv(const ConstIoSlice* slices, size_t count);
    bool wait_readable(int timeout_ms);
    bool wait_writable(int timeout_ms);

    bool is_open() const;
    IoStatus last_status() const { return last_status_; }
    int last_error() const { return last_error_; }
    void close();

    intptr_t native_handle() const { return fd_; }

private:
    intptr_t fd_ = -1;
    IoStatus last_status_ = IoStatus::OK;
    int last_error_ = 0;
};

}  // namespace te
