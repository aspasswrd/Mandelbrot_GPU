#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <thread>
#include <atomic>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_ITER = 800;

std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> colorTable(MAX_ITER + 1);

float offsetX = -0.705922586560551705765;
float offsetY = -0.267652025962102419929;
float zoom = 0.5;
bool needsRedraw = false;

cl::Context context;
cl::Program program;
cl::CommandQueue queue;
cl::Kernel kernel;

std::atomic<bool> isGenerating(false);

void initColorTable() {
    for (int iter = 0; iter <= MAX_ITER; ++iter) {
        long double t = static_cast<long double>(iter) / MAX_ITER;
        uint8_t r = static_cast<uint8_t>(9 * (1 - t) * t * t * t * 255);
        uint8_t g = static_cast<uint8_t>(15 * (1 - t) * (1 - t) * t * t * 255);
        uint8_t b = static_cast<uint8_t>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
        colorTable[iter] = {r, g, b};
    }
}

void initOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!\n";
        exit(1);
    }

    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No GPU devices found!\n";
        exit(1);
    }

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, devices[0]);

    const char* kernelSource = R"(
    __kernel void mandelbrot(__global uchar3* output,
                            const float offsetX,
                            const float offsetY,
                            const float zoom,
                            const int max_iter) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int width = get_global_size(0);
        int height = get_global_size(1);

        float scaleX = 3.5 / width / zoom;
        float scaleY = 2.0 / height / zoom;

        float cx = (x - width/2) * scaleX + offsetX;
        float cy = (y - height/2) * scaleY + offsetY;

        float zx = 0.0, zy = 0.0;
        int iter = 0;

        while (iter < max_iter) {
            float zx2 = zx * zx;
            float zy2 = zy * zy;
            if (zx2 + zy2 > 4.0) break;

            float tmp = zx2 - zy2 + cx;
            zy = 2.0 * zx * zy + cy;
            zx = tmp;
            iter++;
        }

        output[y * width + x] = (uchar3)(iter % 256, iter % 256, iter % 256);
    }
    )";

    cl::Program::Sources sources;
    sources.push_back({kernelSource, strlen(kernelSource)});
    program = cl::Program(context, sources);
    if (program.build() != CL_SUCCESS) {
        std::cout << "Build error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << "\n";
        exit(1);
    }
    kernel = cl::Kernel(program, "mandelbrot");
}

void generateMandelbrot(std::vector<uint8_t>& image) {
    cl::Buffer outputBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar3) * WIDTH * HEIGHT);

    kernel.setArg(0, outputBuf);
    kernel.setArg(1, offsetX);
    kernel.setArg(2, offsetY);
    kernel.setArg(3, zoom);
    kernel.setArg(4, MAX_ITER);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(WIDTH, HEIGHT));

    cl_uchar3* output = (cl_uchar3*)queue.enqueueMapBuffer(outputBuf, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uchar3) * WIDTH * HEIGHT);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            auto [r, g, b] = colorTable[output[idx].s0];
            image[(y * WIDTH + x) * 3] = r;
            image[(y * WIDTH + x) * 3 + 1] = g;
            image[(y * WIDTH + x) * 3 + 2] = b;
        }
    }

    queue.enqueueUnmapMemObject(outputBuf, output);
    queue.finish();
}


void generateMandelbrotAsync(std::vector<uint8_t>& image) {
    if (isGenerating) return;
    isGenerating = true;
    std::thread([&image]() {
        generateMandelbrot(image);
        isGenerating = false;
    }).detach();
}

int main() {
    initColorTable();
    initOpenCL();

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init error: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Mandelbrot GPU", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    std::vector<uint8_t> image(WIDTH * HEIGHT * 3);
    generateMandelbrotAsync(image);

    bool allZero = true;
    for (int i = 0; i < WIDTH * HEIGHT * 3; ++i) {
        if (image[i] != 0) {
            allZero = false;
            break;
        }
    }
    std::cout << "Изображение пустое: " << (allZero ? "Да" : "Нет") << "\n";

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_w:
                        offsetY -= (0.1f / zoom);
                    needsRedraw = true;
                    break;
                    case SDLK_s:
                        offsetY += (0.1f / zoom);
                    needsRedraw = true;
                    break;
                    case SDLK_a:
                        offsetX -= (0.1f / zoom);
                    needsRedraw = true;
                    break;
                    case SDLK_d:
                        offsetX += (0.1f / zoom);
                    needsRedraw = true;
                    break;
                    case SDLK_e:
                        zoom *= 1.05L;
                    needsRedraw = true;
                    break;
                    case SDLK_q:
                        zoom /= 1.05L;
                    needsRedraw = true;
                    break;
                }
            }
        }

        if (needsRedraw) {
            generateMandelbrotAsync(image);
            needsRedraw = false;
        }

        SDL_UpdateTexture(texture, nullptr, image.data(), WIDTH * 3);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
