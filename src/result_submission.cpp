// CPU Benchmark - Result Submission Module Implementation

#include "result_submission.hpp"
#include "cpu_capabilities.hpp"
#include "version.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <array>
#include <cstdint>

// Platform-specific includes for sockets
#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <poll.h>
    #include <errno.h>
#endif

// Forward declarations of internal functions
static bool prompt_user_consent();
static SubmissionResult http_post(const std::string& url, const std::string& json_body);

// Simple SHA-256 implementation (minimal, for integrity tokens)
namespace sha256 {

// SHA-256 constants
static constexpr std::array<uint32_t, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sig0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline uint32_t sig1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline uint32_t gam0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline uint32_t gam1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

std::string hash(const std::string& input) {
    // Initial hash values
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Pre-processing: adding padding bits
    std::vector<uint8_t> msg(input.begin(), input.end());
    uint64_t bit_len = msg.size() * 8;
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0x00);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xff));
    
    // Process each 512-bit chunk
    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(msg[chunk + i*4]) << 24) |
                   (static_cast<uint32_t>(msg[chunk + i*4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[chunk + i*4 + 2]) << 8) |
                   static_cast<uint32_t>(msg[chunk + i*4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            w[i] = gam1(w[i-2]) + w[i-7] + gam0(w[i-15]) + w[i-16];
        }
        
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = hh + sig1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = sig0(a) + maj(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    
    // Produce final hash string
    std::ostringstream oss;
    for (int i = 0; i < 8; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(8) << h[i];
    }
    return oss.str();
}

} // namespace sha256


// Helper to escape JSON strings
static std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(c);
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

// Build JSON payload from benchmark data (with optional nickname and mode override)
std::string result_submission::build_payload(const BenchmarkResult& result, const Config& config, const CpuInfo& cpu_info, const std::string& nickname, const std::string& mode_override) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    // Generate integrity data (pass mode_override to ensure token matches payload)
    std::string hw_fingerprint = result_submission::generate_hardware_fingerprint(cpu_info);
    std::string integrity_token = result_submission::generate_integrity_token(result, config, hw_fingerprint, mode_override);
    
    // Get OS version once
    std::string os_version = get_os_version();
    
    oss << "{";
    
    // Nickname
    oss << "\"nickname\":\"" << json_escape(nickname.empty() ? "Anonymous" : nickname) << "\",";
    
    // Benchmark section
    oss << "\"benchmark\":{";
    // Use mode_override if provided, otherwise use config.mode
    std::string mode_str = mode_override.empty() ? mode_to_string(config.mode) : mode_override;
    oss << "\"mode\":\"" << mode_str << "\",";
    oss << "\"size\":{\"Nx\":" << config.size.Nx << ",\"Ny\":" << config.size.Ny << ",\"Nz\":" << config.size.Nz << "},";
    oss << "\"precision\":\"" << precision_to_string(config.precision) << "\",";
    oss << "\"threads\":" << config.threads << ",";
    oss << "\"repeats\":" << config.repeats << ",";
    oss << "\"iterations\":" << result.iterations;
    oss << "},";
    
    // Results section
    oss << "\"results\":{";
    oss << "\"time_avg_sec\":" << result.time_avg_sec << ",";
    oss << "\"time_min_sec\":" << result.time_min_sec << ",";
    oss << "\"time_stddev_sec\":" << result.time_stddev_sec << ",";
    oss << "\"gflops_avg\":" << result.gflops_avg << ",";
    oss << "\"gflops_max\":" << result.gflops_max << ",";
    oss << "\"total_flops\":" << result.total_flops;
    oss << "},";
    
    // CPU section
    oss << "\"cpu\":{";
    oss << "\"arch\":\"" << json_escape(cpu_info.arch) << "\",";
    oss << "\"vendor\":\"" << json_escape(cpu_info.vendor) << "\",";
    oss << "\"model\":\"" << json_escape(cpu_info.model) << "\",";
    oss << "\"logical_cores\":" << cpu_info.logical_cores << ",";
    oss << "\"physical_cores\":" << cpu_info.physical_cores << ",";
    oss << "\"socket_count\":" << cpu_info.socket_count << ",";
    oss << "\"selected_socket\":" << config.selected_socket;
    oss << "},";
    
    // SIMD section (includes both x86 and ARM capabilities)
    const auto& caps = CpuCapabilities::get();
    oss << "\"simd\":{";
    oss << "\"sse2\":" << (caps.has_sse2 ? "true" : "false") << ",";
    oss << "\"avx\":" << (caps.has_avx ? "true" : "false") << ",";
    oss << "\"avx2\":" << (caps.has_avx2 ? "true" : "false") << ",";
    oss << "\"avx512f\":" << (caps.has_avx512f ? "true" : "false") << ",";
    oss << "\"neon\":" << (caps.has_arm_neon ? "true" : "false") << ",";
    oss << "\"neon_fp16\":" << (caps.has_arm_neon_fp16 ? "true" : "false");
    oss << "},";
    
    // Cache section
    oss << "\"cache\":{";
    oss << "\"l1_data_kb\":" << (cpu_info.cache.l1_available ? static_cast<int>(cpu_info.cache.l1_data_size / 1024) : 0) << ",";
    oss << "\"l2_kb\":" << (cpu_info.cache.l2_available ? static_cast<int>(cpu_info.cache.l2_size / 1024) : 0) << ",";
    oss << "\"l3_kb\":" << (cpu_info.cache.l3_available ? static_cast<int>(cpu_info.cache.l3_size / 1024) : 0);
    oss << "},";
    
    // OS section
    oss << "\"os\":{";
#ifdef _WIN32
    oss << "\"name\":\"Windows\",";
    oss << "\"type\":\"windows\",";
#elif defined(__APPLE__)
    oss << "\"name\":\"macOS\",";
    oss << "\"type\":\"darwin\",";
#elif defined(__linux__)
    oss << "\"name\":\"Linux\",";
    oss << "\"type\":\"linux\",";
#elif defined(__FreeBSD__)
    oss << "\"name\":\"FreeBSD\",";
    oss << "\"type\":\"freebsd\",";
#else
    oss << "\"name\":\"Unknown\",";
    oss << "\"type\":\"unknown\",";
#endif
    oss << "\"version\":\"" << json_escape(os_version) << "\"";
    oss << "},";
    
    // Integrity section
    oss << "\"integrity\":{";
    oss << "\"hardware_fingerprint\":\"sha256:" << hw_fingerprint << "\",";
    oss << "\"token\":\"sha256:" << integrity_token << "\"";
    oss << "},";
    
    // Client version
    oss << "\"client_version\":\"" << version::get_version_string() << "\"";
    
    oss << "}";
    
    return oss.str();
}


// Generate hardware fingerprint from CPU characteristics
std::string result_submission::generate_hardware_fingerprint(const CpuInfo& cpu_info) {
    std::ostringstream oss;
    oss << cpu_info.vendor << "|";
    oss << cpu_info.model << "|";
    oss << cpu_info.arch << "|";
    oss << cpu_info.logical_cores << "|";
    oss << cpu_info.physical_cores;
    return sha256::hash(oss.str());
}

// Generate integrity token from execution parameters
std::string result_submission::generate_integrity_token(const BenchmarkResult& result, const Config& config, const std::string& hw_fingerprint, const std::string& mode_override) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    // Include benchmark parameters (use mode_override if provided)
    std::string mode_str = mode_override.empty() ? mode_to_string(config.mode) : mode_override;
    oss << mode_str << "|";
    oss << config.size.Nx << "|" << config.size.Ny << "|" << config.size.Nz << "|";
    oss << precision_to_string(config.precision) << "|";
    oss << config.threads << "|";
    oss << config.repeats << "|";
    
    // Include results
    oss << result.time_avg_sec << "|";
    oss << result.time_min_sec << "|";
    oss << result.gflops_avg << "|";
    oss << result.gflops_max << "|";
    oss << result.total_flops << "|";
    
    // Include hardware fingerprint
    oss << hw_fingerprint;
    
    return sha256::hash(oss.str());
}


// Parse URL into host, port, and path
static bool parse_url(const std::string& url, std::string& host, int& port, std::string& path) {
    // Expected format: http://host:port/path or http://host/path
    size_t proto_end = url.find("://");
    if (proto_end == std::string::npos) return false;
    
    size_t host_start = proto_end + 3;
    size_t path_start = url.find('/', host_start);
    if (path_start == std::string::npos) {
        path = "/";
        path_start = url.length();
    } else {
        path = url.substr(path_start);
    }
    
    std::string host_port = url.substr(host_start, path_start - host_start);
    size_t colon = host_port.find(':');
    if (colon == std::string::npos) {
        host = host_port;
        port = 80;
    } else {
        host = host_port.substr(0, colon);
        port = std::stoi(host_port.substr(colon + 1));
    }
    
    return true;
}

#ifdef _WIN32
// Windows socket implementation
static SubmissionResult http_post(const std::string& url, const std::string& json_body) {
    SubmissionResult result;
    result.success = false;
    
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        result.error_message = "Failed to initialize Winsock";
        return result;
    }
    
    std::string host, path;
    int port;
    if (!parse_url(url, host, port, path)) {
        result.error_message = "Invalid URL format";
        WSACleanup();
        return result;
    }
    
    // Resolve hostname
    struct addrinfo hints = {}, *addr_result = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &addr_result) != 0) {
        result.error_message = "Failed to resolve hostname: " + host;
        WSACleanup();
        return result;
    }
    
    // Create socket
    SOCKET sock = socket(addr_result->ai_family, addr_result->ai_socktype, addr_result->ai_protocol);
    if (sock == INVALID_SOCKET) {
        result.error_message = "Failed to create socket";
        freeaddrinfo(addr_result);
        WSACleanup();
        return result;
    }
    
    // Set timeout (10 seconds)
    DWORD timeout = 10000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    
    // Connect
    if (connect(sock, addr_result->ai_addr, static_cast<int>(addr_result->ai_addrlen)) == SOCKET_ERROR) {
        result.error_message = "Failed to connect to server";
        closesocket(sock);
        freeaddrinfo(addr_result);
        WSACleanup();
        return result;
    }
    freeaddrinfo(addr_result);
    
    // Build HTTP request
    std::ostringstream request;
    std::string api_path = (path == "/" || path.empty()) ? "/api/submit" : path + "/api/submit";
    request << "POST " << api_path << " HTTP/1.1\r\n";
    request << "Host: " << host << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << json_body.length() << "\r\n";
    request << "Connection: close\r\n";
    request << "\r\n";
    request << json_body;
    
    std::string req_str = request.str();
    if (send(sock, req_str.c_str(), static_cast<int>(req_str.length()), 0) == SOCKET_ERROR) {
        result.error_message = "Failed to send request";
        closesocket(sock);
        WSACleanup();
        return result;
    }
    
    // Receive response
    std::string response;
    char buffer[4096];
    int bytes_received;
    while ((bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_received] = '\0';
        response += buffer;
    }
    
    closesocket(sock);
    WSACleanup();
    
    // Parse response
    size_t status_end = response.find("\r\n");
    if (status_end == std::string::npos) {
        result.error_message = "Invalid server response";
        return result;
    }
    
    std::string status_line = response.substr(0, status_end);
    int status_code = 0;
    if (status_line.length() > 12) {
        status_code = std::stoi(status_line.substr(9, 3));
    }
    
    // Find response body
    size_t body_start = response.find("\r\n\r\n");
    std::string body = (body_start != std::string::npos) ? response.substr(body_start + 4) : "";
    
    if (status_code == 201) {
        result.success = true;
        // Extract ID from JSON response
        size_t id_pos = body.find("\"id\"");
        if (id_pos != std::string::npos) {
            size_t id_start = body.find("\"", id_pos + 4);
            size_t id_end = body.find("\"", id_start + 1);
            if (id_start != std::string::npos && id_end != std::string::npos) {
                result.result_id = body.substr(id_start + 1, id_end - id_start - 1);
            }
        }
    } else if (status_code >= 400 && status_code < 500) {
        // Extract error message from JSON
        size_t err_pos = body.find("\"error\"");
        if (err_pos != std::string::npos) {
            size_t err_start = body.find("\"", err_pos + 7);
            size_t err_end = body.find("\"", err_start + 1);
            if (err_start != std::string::npos && err_end != std::string::npos) {
                result.error_message = body.substr(err_start + 1, err_end - err_start - 1);
            }
        } else {
            result.error_message = "Server rejected submission (HTTP " + std::to_string(status_code) + ")";
        }
    } else if (status_code >= 500) {
        result.error_message = "Server error, please try again later";
    } else {
        result.error_message = "Unexpected response (HTTP " + std::to_string(status_code) + ")";
    }
    
    return result;
}


#else
// POSIX socket implementation
static SubmissionResult http_post(const std::string& url, const std::string& json_body) {
    SubmissionResult result;
    result.success = false;
    
    std::string host, path;
    int port;
    if (!parse_url(url, host, port, path)) {
        result.error_message = "Invalid URL format";
        return result;
    }
    
    // Resolve hostname
    struct addrinfo hints = {}, *addr_result = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &addr_result) != 0) {
        result.error_message = "Failed to resolve hostname: " + host;
        return result;
    }
    
    // Create socket
    int sock = socket(addr_result->ai_family, addr_result->ai_socktype, addr_result->ai_protocol);
    if (sock < 0) {
        result.error_message = "Failed to create socket";
        freeaddrinfo(addr_result);
        return result;
    }
    
    // Set timeout (10 seconds)
    struct timeval timeout;
    timeout.tv_sec = 10;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    // Connect
    if (connect(sock, addr_result->ai_addr, addr_result->ai_addrlen) < 0) {
        result.error_message = "Failed to connect to server";
        close(sock);
        freeaddrinfo(addr_result);
        return result;
    }
    freeaddrinfo(addr_result);
    
    // Build HTTP request
    std::ostringstream request;
    std::string api_path = (path == "/" || path.empty()) ? "/api/submit" : path + "/api/submit";
    request << "POST " << api_path << " HTTP/1.1\r\n";
    request << "Host: " << host << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << json_body.length() << "\r\n";
    request << "Connection: close\r\n";
    request << "\r\n";
    request << json_body;
    
    std::string req_str = request.str();
    if (send(sock, req_str.c_str(), req_str.length(), 0) < 0) {
        result.error_message = "Failed to send request";
        close(sock);
        return result;
    }
    
    // Receive response
    std::string response;
    char buffer[4096];
    ssize_t bytes_received;
    while ((bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_received] = '\0';
        response += buffer;
    }
    
    close(sock);
    
    // Parse response
    size_t status_end = response.find("\r\n");
    if (status_end == std::string::npos) {
        result.error_message = "Invalid server response";
        return result;
    }
    
    std::string status_line = response.substr(0, status_end);
    int status_code = 0;
    if (status_line.length() > 12) {
        status_code = std::stoi(status_line.substr(9, 3));
    }
    
    // Find response body
    size_t body_start = response.find("\r\n\r\n");
    std::string body = (body_start != std::string::npos) ? response.substr(body_start + 4) : "";
    
    if (status_code == 201) {
        result.success = true;
        // Extract ID from JSON response
        size_t id_pos = body.find("\"id\"");
        if (id_pos != std::string::npos) {
            size_t id_start = body.find("\"", id_pos + 4);
            size_t id_end = body.find("\"", id_start + 1);
            if (id_start != std::string::npos && id_end != std::string::npos) {
                result.result_id = body.substr(id_start + 1, id_end - id_start - 1);
            }
        }
    } else if (status_code >= 400 && status_code < 500) {
        // Extract error message from JSON
        size_t err_pos = body.find("\"error\"");
        if (err_pos != std::string::npos) {
            size_t err_start = body.find("\"", err_pos + 7);
            size_t err_end = body.find("\"", err_start + 1);
            if (err_start != std::string::npos && err_end != std::string::npos) {
                result.error_message = body.substr(err_start + 1, err_end - err_start - 1);
            }
        } else {
            result.error_message = "Server rejected submission (HTTP " + std::to_string(status_code) + ")";
        }
    } else if (status_code >= 500) {
        result.error_message = "Server error, please try again later";
    } else {
        result.error_message = "Unexpected response (HTTP " + std::to_string(status_code) + ")";
    }
    
    return result;
}
#endif


// Prompt user for submission consent
static bool prompt_user_consent() {
    std::cout << "\nWould you like to submit your results to the benchmark server? (y/n): ";
    std::cout.flush();
    
    std::string input;
    if (!std::getline(std::cin, input)) {
        return false;
    }
    
    // Trim whitespace
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return false;
    size_t end = input.find_last_not_of(" \t\r\n");
    input = input.substr(start, end - start + 1);
    
    return (input == "y" || input == "Y" || input == "yes" || input == "Yes" || input == "YES");
}

// Prompt user for nickname
static std::string prompt_nickname() {
    std::cout << "Enter your nickname (press Enter for Anonymous): ";
    std::cout.flush();
    
    std::string input;
    if (!std::getline(std::cin, input)) {
        return "Anonymous";
    }
    
    // Trim whitespace
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "Anonymous";
    size_t end = input.find_last_not_of(" \t\r\n");
    input = input.substr(start, end - start + 1);
    
    return input.empty() ? "Anonymous" : input;
}

// Main entry point - prompts user and submits if confirmed
bool submit_results_interactive(
    const BenchmarkResult& result,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url,
    const std::string& mode_override)
{
    // Prompt user for consent
    if (!prompt_user_consent()) {
        return false;
    }
    
    // Prompt for nickname
    std::string nickname = prompt_nickname();
    
    // Display status
    std::cout << "Submitting results as \"" << nickname << "\"..." << std::flush;
    
    // Build payload with nickname and optional mode override
    std::string payload = result_submission::build_payload(result, config, cpu_info, nickname, mode_override);
    
    // Submit to server
    SubmissionResult submission = http_post(server_url, payload);
    
    // Display result
    if (submission.success) {
        std::cout << " Done!\n";
        std::cout << "Result ID: " << submission.result_id << "\n";
        return true;
    } else {
        std::cout << " Failed!\n";
        std::cout << "Error: " << submission.error_message << "\n";
        return false;
    }
}

// Build JSON payload for compute benchmark results
static std::string build_compute_payload(const ComputeSubmissionData& compute_data, const Config& config, 
                                         const CpuInfo& cpu_info, const std::string& nickname) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    // Generate integrity data based on compute results
    std::string hw_fingerprint = result_submission::generate_hardware_fingerprint(cpu_info);
    
    // Create a simple hash for compute integrity
    std::ostringstream token_data;
    token_data << std::fixed << std::setprecision(9);
    token_data << "compute|";
    token_data << config.size.Nx << "|" << config.size.Ny << "|" << config.size.Nz << "|";
    token_data << compute_data.mt_threads << "|";
    token_data << compute_data.st_gflops << "|";
    token_data << compute_data.mt_gflops << "|";
    token_data << compute_data.st_score << "|";
    token_data << compute_data.mt_score << "|";
    token_data << hw_fingerprint;
    std::string integrity_token = sha256::hash(token_data.str());
    
    oss << "{";
    
    // Nickname
    oss << "\"nickname\":\"" << json_escape(nickname.empty() ? "Anonymous" : nickname) << "\",";
    
    // Benchmark section
    oss << "\"benchmark\":{";
    oss << "\"mode\":\"compute\",";
    oss << "\"size\":{\"Nx\":" << config.size.Nx << ",\"Ny\":" << config.size.Ny << ",\"Nz\":" << config.size.Nz << "},";
    oss << "\"precision\":\"float\",";
    oss << "\"threads\":" << compute_data.mt_threads << ",";
    oss << "\"repeats\":" << config.repeats << ",";
    oss << "\"iterations\":1";
    oss << "},";
    
    // Results section with ST/MT data
    oss << "\"results\":{";
    oss << "\"time_avg_sec\":" << compute_data.mt_time_sec << ",";
    oss << "\"time_min_sec\":" << compute_data.mt_time_sec << ",";
    oss << "\"time_stddev_sec\":0.0,";
    oss << "\"gflops_avg\":" << compute_data.mt_gflops << ",";
    oss << "\"gflops_max\":" << compute_data.mt_gflops << ",";
    oss << "\"total_flops\":" << static_cast<uint64_t>(compute_data.mt_gflops * 1e9 * compute_data.mt_time_sec) << ",";
    // Compute-specific fields
    oss << "\"st_time_sec\":" << compute_data.st_time_sec << ",";
    oss << "\"st_gflops\":" << compute_data.st_gflops << ",";
    oss << "\"st_score\":" << compute_data.st_score << ",";
    oss << "\"st_threads\":" << compute_data.st_threads << ",";
    oss << "\"mt_time_sec\":" << compute_data.mt_time_sec << ",";
    oss << "\"mt_gflops\":" << compute_data.mt_gflops << ",";
    oss << "\"mt_score\":" << compute_data.mt_score << ",";
    oss << "\"mt_threads\":" << compute_data.mt_threads << ",";
    oss << "\"overall_score\":" << compute_data.overall_score << ",";
    oss << "\"simd_level\":\"" << json_escape(compute_data.simd_level) << "\"";
    // Frequency data inside results section
    if (compute_data.frequency.available) {
        oss << ",\"frequency\":{";
        oss << "\"min_mhz\":" << compute_data.frequency.min_mhz << ",";
        oss << "\"max_mhz\":" << compute_data.frequency.max_mhz << ",";
        oss << "\"avg_mhz\":" << compute_data.frequency.avg_mhz << ",";
        oss << "\"available\":true";
        oss << "}";
    }
    oss << "},";
    
    // CPU section
    oss << "\"cpu\":{";
    oss << "\"arch\":\"" << json_escape(cpu_info.arch) << "\",";
    oss << "\"vendor\":\"" << json_escape(cpu_info.vendor) << "\",";
    oss << "\"model\":\"" << json_escape(cpu_info.model) << "\",";
    oss << "\"logical_cores\":" << cpu_info.logical_cores << ",";
    oss << "\"physical_cores\":" << cpu_info.physical_cores << ",";
    oss << "\"socket_count\":" << compute_data.socket_count << ",";
    oss << "\"selected_socket\":" << compute_data.selected_socket << ",";
    oss << "\"instructions\":\"" << json_escape(get_cpu_instructions_string()) << "\"";
    oss << "},";
    
    // SIMD section (includes both x86 and ARM capabilities)
    const auto& caps = CpuCapabilities::get();
    oss << "\"simd\":{";
    oss << "\"sse2\":" << (caps.has_sse2 ? "true" : "false") << ",";
    oss << "\"avx\":" << (caps.has_avx ? "true" : "false") << ",";
    oss << "\"avx2\":" << (caps.has_avx2 ? "true" : "false") << ",";
    oss << "\"avx512f\":" << (caps.has_avx512f ? "true" : "false") << ",";
    oss << "\"neon\":" << (caps.has_arm_neon ? "true" : "false") << ",";
    oss << "\"neon_fp16\":" << (caps.has_arm_neon_fp16 ? "true" : "false");
    oss << "},";
    
    // Cache section
    oss << "\"cache\":{";
    oss << "\"l1_data_kb\":" << (cpu_info.cache.l1_available ? static_cast<int>(cpu_info.cache.l1_data_size / 1024) : 0) << ",";
    oss << "\"l2_kb\":" << (cpu_info.cache.l2_available ? static_cast<int>(cpu_info.cache.l2_size / 1024) : 0) << ",";
    oss << "\"l3_kb\":" << (cpu_info.cache.l3_available ? static_cast<int>(cpu_info.cache.l3_size / 1024) : 0);
    oss << "},";
    
    // OS section
    std::string os_version = compute_data.os_version.empty() ? get_os_version() : compute_data.os_version;
    oss << "\"os\":{";
#ifdef _WIN32
    oss << "\"name\":\"Windows\",";
    oss << "\"type\":\"windows\",";
#elif defined(__APPLE__)
    oss << "\"name\":\"macOS\",";
    oss << "\"type\":\"darwin\",";
#elif defined(__linux__)
    oss << "\"name\":\"Linux\",";
    oss << "\"type\":\"linux\",";
#elif defined(__FreeBSD__)
    oss << "\"name\":\"FreeBSD\",";
    oss << "\"type\":\"freebsd\",";
#else
    oss << "\"name\":\"Unknown\",";
    oss << "\"type\":\"unknown\",";
#endif
    oss << "\"version\":\"" << json_escape(os_version) << "\"";
    oss << "},";
    
    // Integrity section
    oss << "\"integrity\":{";
    oss << "\"hardware_fingerprint\":\"sha256:" << hw_fingerprint << "\",";
    oss << "\"token\":\"sha256:" << integrity_token << "\"";
    oss << "},";
    
    // Client version
    oss << "\"client_version\":\"" << version::get_version_string() << "\"";
    
    // Session ID (for linking results from same benchmark run)
    if (!compute_data.session_id.empty()) {
        oss << ",\"session_id\":\"" << json_escape(compute_data.session_id) << "\"";
    }
    
    oss << "}";
    
    return oss.str();
}

// Submit compute benchmark results (with ST/MT scores)
bool submit_compute_results_interactive(
    const ComputeSubmissionData& compute_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url)
{
    // Prompt user for consent
    if (!prompt_user_consent()) {
        return false;
    }
    
    // Prompt for nickname
    std::string nickname = prompt_nickname();
    
    // Display status
    std::cout << "Submitting compute results as \"" << nickname << "\"..." << std::flush;
    
    // Build payload with compute-specific data
    std::string payload = build_compute_payload(compute_data, config, cpu_info, nickname);
    
    // Submit to server
    SubmissionResult submission = http_post(server_url, payload);
    
    // Display result
    if (submission.success) {
        std::cout << " Done!\n";
        std::cout << "Result ID: " << submission.result_id << "\n";
        return true;
    } else {
        std::cout << " Failed!\n";
        std::cout << "Error: " << submission.error_message << "\n";
        return false;
    }
}

// Submit compute benchmark results (non-interactive version)
bool submit_compute_results(
    const ComputeSubmissionData& compute_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& nickname,
    const std::string& server_url)
{
    // Build payload with compute-specific data
    std::string payload = build_compute_payload(compute_data, config, cpu_info, nickname);
    
    // Submit to server
    SubmissionResult submission = http_post(server_url, payload);
    
    return submission.success;
}

// Build JSON payload for precision=all benchmark results (all precision types)
static std::string build_precision_all_payload(const PrecisionAllSubmissionData& precision_data, const Config& config, 
                                               const CpuInfo& cpu_info, const std::string& nickname) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    // Generate integrity data based on all precision results
    std::string hw_fingerprint = result_submission::generate_hardware_fingerprint(cpu_info);
    
    // Create a combined hash for integrity from all results
    std::ostringstream token_data;
    token_data << std::fixed << std::setprecision(9);
    token_data << "precision_all|";
    token_data << config.size.Nx << "|" << config.size.Ny << "|" << config.size.Nz << "|";
    token_data << config.threads << "|";
    for (const auto& r : precision_data.results) {
        token_data << r.precision_name << "|";
        token_data << r.gflops_avg << "|";
        token_data << r.time_min_sec << "|";
    }
    token_data << hw_fingerprint;
    std::string integrity_token = sha256::hash(token_data.str());
    
    oss << "{";
    
    // Nickname
    oss << "\"nickname\":\"" << json_escape(nickname.empty() ? "Anonymous" : nickname) << "\",";
    
    // Benchmark section
    oss << "\"benchmark\":{";
    oss << "\"mode\":\"precision_all\",";
    oss << "\"size\":{\"Nx\":" << config.size.Nx << ",\"Ny\":" << config.size.Ny << ",\"Nz\":" << config.size.Nz << "},";
    oss << "\"precision\":\"all\",";
    oss << "\"threads\":" << config.threads << ",";
    oss << "\"repeats\":" << config.repeats << ",";
    oss << "\"iterations\":1";
    oss << "},";
    
    // Results section - array of all precision results
    oss << "\"results\":{";
    oss << "\"precision_results\":[";
    for (size_t i = 0; i < precision_data.results.size(); ++i) {
        const auto& r = precision_data.results[i];
        if (i > 0) oss << ",";
        oss << "{";
        oss << "\"precision\":\"" << json_escape(r.precision_name) << "\",";
        if (r.precision_name == "fp16") {
            oss << "\"fp16_mode\":\"" << json_escape(r.fp16_mode) << "\",";
        }
        oss << "\"bytes_per_element\":" << r.bytes_per_element << ",";
        oss << "\"is_integer\":" << (r.is_integer ? "true" : "false") << ",";
        oss << "\"is_emulated\":" << (r.is_emulated ? "true" : "false") << ",";
        oss << "\"time_min_sec\":" << r.time_min_sec << ",";
        oss << "\"time_avg_sec\":" << r.time_avg_sec << ",";
        oss << "\"time_stddev_sec\":" << r.time_stddev_sec << ",";
        oss << "\"gflops_avg\":" << r.gflops_avg << ",";
        oss << "\"gflops_max\":" << r.gflops_max << ",";
        oss << "\"total_flops\":" << r.total_flops << ",";
        oss << "\"iterations\":" << r.iterations;
        oss << "}";
    }
    oss << "]";
    oss << "},";
    
    // CPU section
    oss << "\"cpu\":{";
    oss << "\"arch\":\"" << json_escape(cpu_info.arch) << "\",";
    oss << "\"vendor\":\"" << json_escape(cpu_info.vendor) << "\",";
    oss << "\"model\":\"" << json_escape(cpu_info.model) << "\",";
    oss << "\"logical_cores\":" << cpu_info.logical_cores << ",";
    oss << "\"physical_cores\":" << cpu_info.physical_cores << ",";
    oss << "\"socket_count\":" << precision_data.socket_count << ",";
    oss << "\"selected_socket\":" << precision_data.selected_socket << ",";
    oss << "\"instructions\":\"" << json_escape(get_cpu_instructions_string()) << "\"";
    oss << "},";
    
    // SIMD section
    const auto& caps = CpuCapabilities::get();
    oss << "\"simd\":{";
    oss << "\"sse2\":" << (caps.has_sse2 ? "true" : "false") << ",";
    oss << "\"avx\":" << (caps.has_avx ? "true" : "false") << ",";
    oss << "\"avx2\":" << (caps.has_avx2 ? "true" : "false") << ",";
    oss << "\"avx512f\":" << (caps.has_avx512f ? "true" : "false") << ",";
    oss << "\"neon\":" << (caps.has_arm_neon ? "true" : "false") << ",";
    oss << "\"neon_fp16\":" << (caps.has_arm_neon_fp16 ? "true" : "false");
    oss << "},";
    
    // Cache section
    oss << "\"cache\":{";
    oss << "\"l1_data_kb\":" << (cpu_info.cache.l1_available ? static_cast<int>(cpu_info.cache.l1_data_size / 1024) : 0) << ",";
    oss << "\"l2_kb\":" << (cpu_info.cache.l2_available ? static_cast<int>(cpu_info.cache.l2_size / 1024) : 0) << ",";
    oss << "\"l3_kb\":" << (cpu_info.cache.l3_available ? static_cast<int>(cpu_info.cache.l3_size / 1024) : 0);
    oss << "},";
    
    // OS section
    std::string os_version = precision_data.os_version.empty() ? get_os_version() : precision_data.os_version;
    oss << "\"os\":{";
#ifdef _WIN32
    oss << "\"name\":\"Windows\",";
    oss << "\"type\":\"windows\",";
#elif defined(__APPLE__)
    oss << "\"name\":\"macOS\",";
    oss << "\"type\":\"darwin\",";
#elif defined(__linux__)
    oss << "\"name\":\"Linux\",";
    oss << "\"type\":\"linux\",";
#elif defined(__FreeBSD__)
    oss << "\"name\":\"FreeBSD\",";
    oss << "\"type\":\"freebsd\",";
#else
    oss << "\"name\":\"Unknown\",";
    oss << "\"type\":\"unknown\",";
#endif
    oss << "\"version\":\"" << json_escape(os_version) << "\"";
    oss << "},";
    
    // Integrity section
    oss << "\"integrity\":{";
    oss << "\"hardware_fingerprint\":\"sha256:" << hw_fingerprint << "\",";
    oss << "\"token\":\"sha256:" << integrity_token << "\"";
    oss << "},";
    
    // Client version
    oss << "\"client_version\":\"" << version::get_version_string() << "\"";
    
    // Session ID (for linking results from same benchmark run)
    if (!precision_data.session_id.empty()) {
        oss << ",\"session_id\":\"" << json_escape(precision_data.session_id) << "\"";
    }
    
    oss << "}";
    
    return oss.str();
}

// Submit precision=all benchmark results (all precision types data)
bool submit_precision_all_results_interactive(
    const PrecisionAllSubmissionData& precision_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url)
{
    // Prompt user for consent
    if (!prompt_user_consent()) {
        return false;
    }
    
    // Prompt for nickname
    std::string nickname = prompt_nickname();
    
    // Display status
    std::cout << "Submitting precision comparison results as \"" << nickname << "\"..." << std::flush;
    
    // Build payload with all precision results
    std::string payload = build_precision_all_payload(precision_data, config, cpu_info, nickname);
    
    // Submit to server
    SubmissionResult submission = http_post(server_url, payload);
    
    // Display result
    if (submission.success) {
        std::cout << " Done!\n";
        std::cout << "Result ID: " << submission.result_id << "\n";
        return true;
    } else {
        std::cout << " Failed!\n";
        std::cout << "Error: " << submission.error_message << "\n";
        return false;
    }
}

// Submit precision=all benchmark results (non-interactive version)
bool submit_precision_all_results(
    const PrecisionAllSubmissionData& precision_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& nickname,
    const std::string& server_url)
{
    // Build payload with all precision results
    std::string payload = build_precision_all_payload(precision_data, config, cpu_info, nickname);
    
    // Submit to server
    SubmissionResult submission = http_post(server_url, payload);
    
    return submission.success;
}


// Build JSON payload for full benchmark suite results
static std::string build_full_benchmark_payload(const FullBenchmarkSubmissionData& full_data, const Config& config, 
                                                const CpuInfo& cpu_info, const std::string& nickname) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    // Generate integrity data
    std::string hw_fingerprint = result_submission::generate_hardware_fingerprint(cpu_info);
    
    // Create a combined hash for integrity from all results
    std::ostringstream token_data;
    token_data << std::fixed << std::setprecision(9);
    token_data << "full_benchmark|";
    // Compute results
    token_data << full_data.compute.st_gflops << "|" << full_data.compute.mt_gflops << "|";
    token_data << full_data.compute.st_score << "|" << full_data.compute.mt_score << "|";
    // Precision results
    for (const auto& r : full_data.precision.results) {
        token_data << r.precision_name << "|" << r.gflops_avg << "|";
    }
    // Mem results
    token_data << full_data.mem.gflops_avg << "|";
    // Stencil results
    token_data << full_data.stencil.gflops_avg << "|";
    // Cache results
    token_data << full_data.cache.gflops_avg << "|";
    token_data << hw_fingerprint;
    std::string integrity_token = sha256::hash(token_data.str());
    
    oss << "{";
    
    // Nickname
    oss << "\"nickname\":\"" << json_escape(nickname.empty() ? "Anonymous" : nickname) << "\",";
    
    // Benchmark section
    oss << "\"benchmark\":{";
    oss << "\"mode\":\"full_benchmark\",";
    oss << "\"size\":{\"Nx\":" << config.size.Nx << ",\"Ny\":" << config.size.Ny << ",\"Nz\":" << config.size.Nz << "},";
    oss << "\"precision\":\"all\",";
    oss << "\"threads\":" << config.threads << ",";
    oss << "\"repeats\":" << config.repeats << ",";
    oss << "\"iterations\":1";
    oss << "},";
    
    // Results section - all test results
    oss << "\"results\":{";
    
    // Compute results
    oss << "\"compute\":{";
    oss << "\"st_time_sec\":" << full_data.compute.st_time_sec << ",";
    oss << "\"st_gflops\":" << full_data.compute.st_gflops << ",";
    oss << "\"st_score\":" << full_data.compute.st_score << ",";
    oss << "\"st_threads\":" << full_data.compute.st_threads << ",";
    oss << "\"mt_time_sec\":" << full_data.compute.mt_time_sec << ",";
    oss << "\"mt_gflops\":" << full_data.compute.mt_gflops << ",";
    oss << "\"mt_score\":" << full_data.compute.mt_score << ",";
    oss << "\"mt_threads\":" << full_data.compute.mt_threads << ",";
    oss << "\"overall_score\":" << full_data.compute.overall_score << ",";
    oss << "\"simd_level\":\"" << json_escape(full_data.compute.simd_level) << "\"";
    oss << "},";
    
    // Precision results
    oss << "\"precision_results\":[";
    for (size_t i = 0; i < full_data.precision.results.size(); ++i) {
        const auto& r = full_data.precision.results[i];
        if (i > 0) oss << ",";
        oss << "{";
        oss << "\"precision\":\"" << json_escape(r.precision_name) << "\",";
        if (r.precision_name == "fp16") {
            oss << "\"fp16_mode\":\"" << json_escape(r.fp16_mode) << "\",";
        }
        oss << "\"bytes_per_element\":" << r.bytes_per_element << ",";
        oss << "\"is_integer\":" << (r.is_integer ? "true" : "false") << ",";
        oss << "\"is_emulated\":" << (r.is_emulated ? "true" : "false") << ",";
        oss << "\"time_min_sec\":" << r.time_min_sec << ",";
        oss << "\"time_avg_sec\":" << r.time_avg_sec << ",";
        oss << "\"gflops_avg\":" << r.gflops_avg << ",";
        oss << "\"gflops_max\":" << r.gflops_max;
        oss << "}";
    }
    oss << "],";
    
    // Mem results
    oss << "\"mem\":{";
    oss << "\"time_avg_sec\":" << full_data.mem.time_avg_sec << ",";
    oss << "\"time_min_sec\":" << full_data.mem.time_min_sec << ",";
    oss << "\"gflops_avg\":" << full_data.mem.gflops_avg << ",";
    oss << "\"gflops_max\":" << full_data.mem.gflops_max << ",";
    oss << "\"bandwidth_gbs\":" << full_data.mem.bandwidth_gbs;
    oss << "},";
    
    // Stencil results
    oss << "\"stencil\":{";
    oss << "\"time_avg_sec\":" << full_data.stencil.time_avg_sec << ",";
    oss << "\"time_min_sec\":" << full_data.stencil.time_min_sec << ",";
    oss << "\"gflops_avg\":" << full_data.stencil.gflops_avg << ",";
    oss << "\"gflops_max\":" << full_data.stencil.gflops_max << ",";
    oss << "\"mlups_avg\":" << full_data.stencil.mlups_avg;
    oss << "},";
    
    // Cache results
    oss << "\"cache\":{";
    oss << "\"time_avg_sec\":" << full_data.cache.time_avg_sec << ",";
    oss << "\"time_min_sec\":" << full_data.cache.time_min_sec << ",";
    oss << "\"gflops_avg\":" << full_data.cache.gflops_avg << ",";
    oss << "\"gflops_max\":" << full_data.cache.gflops_max << ",";
    oss << "\"bandwidth_gbs\":" << full_data.cache.bandwidth_gbs;
    oss << "}";
    
    oss << "},";
    
    // CPU section
    oss << "\"cpu\":{";
    oss << "\"arch\":\"" << json_escape(cpu_info.arch) << "\",";
    oss << "\"vendor\":\"" << json_escape(cpu_info.vendor) << "\",";
    oss << "\"model\":\"" << json_escape(cpu_info.model) << "\",";
    oss << "\"logical_cores\":" << cpu_info.logical_cores << ",";
    oss << "\"physical_cores\":" << cpu_info.physical_cores << ",";
    oss << "\"socket_count\":" << full_data.socket_count << ",";
    oss << "\"selected_socket\":" << full_data.selected_socket;
    oss << "},";
    
    // SIMD section
    const auto& caps = CpuCapabilities::get();
    oss << "\"simd\":{";
    oss << "\"sse2\":" << (caps.has_sse2 ? "true" : "false") << ",";
    oss << "\"avx\":" << (caps.has_avx ? "true" : "false") << ",";
    oss << "\"avx2\":" << (caps.has_avx2 ? "true" : "false") << ",";
    oss << "\"avx512f\":" << (caps.has_avx512f ? "true" : "false") << ",";
    oss << "\"neon\":" << (caps.has_arm_neon ? "true" : "false") << ",";
    oss << "\"neon_fp16\":" << (caps.has_arm_neon_fp16 ? "true" : "false");
    oss << "},";
    
    // Cache section
    oss << "\"cache_info\":{";
    oss << "\"l1_data_kb\":" << (cpu_info.cache.l1_available ? static_cast<int>(cpu_info.cache.l1_data_size / 1024) : 0) << ",";
    oss << "\"l2_kb\":" << (cpu_info.cache.l2_available ? static_cast<int>(cpu_info.cache.l2_size / 1024) : 0) << ",";
    oss << "\"l3_kb\":" << (cpu_info.cache.l3_available ? static_cast<int>(cpu_info.cache.l3_size / 1024) : 0);
    oss << "},";
    
    // OS section
    std::string os_version = full_data.os_version.empty() ? get_os_version() : full_data.os_version;
    oss << "\"os\":{";
#ifdef _WIN32
    oss << "\"name\":\"Windows\",";
    oss << "\"type\":\"windows\",";
#elif defined(__APPLE__)
    oss << "\"name\":\"macOS\",";
    oss << "\"type\":\"darwin\",";
#elif defined(__linux__)
    oss << "\"name\":\"Linux\",";
    oss << "\"type\":\"linux\",";
#elif defined(__FreeBSD__)
    oss << "\"name\":\"FreeBSD\",";
    oss << "\"type\":\"freebsd\",";
#else
    oss << "\"name\":\"Unknown\",";
    oss << "\"type\":\"unknown\",";
#endif
    oss << "\"version\":\"" << json_escape(os_version) << "\"";
    oss << "},";
    
    // Integrity section
    oss << "\"integrity\":{";
    oss << "\"hardware_fingerprint\":\"sha256:" << hw_fingerprint << "\",";
    oss << "\"token\":\"sha256:" << integrity_token << "\"";
    oss << "},";
    
    // Client version
    oss << "\"client_version\":\"" << version::get_version_string() << "\"";
    
    oss << "}";
    
    return oss.str();
}

// Submit full benchmark suite results (all tests combined)
bool submit_full_benchmark_results_interactive(
    const FullBenchmarkSubmissionData& full_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url)
{
    // Prompt user for consent
    if (!prompt_user_consent()) {
        return false;
    }
    
    // Prompt for nickname
    std::string nickname = prompt_nickname();
    
    // Display status
    std::cout << "Submitting full benchmark results as \"" << nickname << "\"..." << std::flush;
    
    // Build payload with all test results
    std::string payload = build_full_benchmark_payload(full_data, config, cpu_info, nickname);
    
    // Submit to server (use /api/submit_full endpoint)
    std::string full_url = server_url;
    // Replace /api/submit with /api/submit_full if present
    size_t pos = full_url.find("/api/submit");
    if (pos != std::string::npos) {
        full_url = full_url.substr(0, pos);
    }
    
    SubmissionResult submission = http_post(full_url + "/api/submit_full", payload);
    
    // Display result
    if (submission.success) {
        std::cout << " Done!\n";
        std::cout << "Result ID: " << submission.result_id << "\n";
        return true;
    } else {
        std::cout << " Failed!\n";
        std::cout << "Error: " << submission.error_message << "\n";
        return false;
    }
}
