/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Demo: verify flagcxGetPeerPointer(mem, offset, team, peer) correctness.
 *
 * Each rank allocates a buffer of nRanks ints, initialized to 0.
 * A kernel uses flagcxGetPeerPointer to obtain peer's buffer pointer
 * and writes myRank into peer's buffer at position [myRank].
 * After sync + D2H copy, each rank checks buf[i] == i for all i.
 *
 * Usage: mpirun -np <nGPUs> ./test_get_peer_pointer [-R 0|1|2]
 *   -R 0: raw (cudaMalloc, no registration)
 *   -R 1: IPC (flagcxMemAlloc + CommRegister)
 *   -R 2: window (flagcxMemAlloc + CommWindowRegister)
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "nvidia_adaptor.h"
#include "tools.h"
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Kernel: each thread writes myRank to peer[tid]'s buffer at offset [myRank]
// Only thread 0 does the work (single-threaded for simplicity)
// ---------------------------------------------------------------------------
__global__ void kernelWriteToPeer(flagcxDevComm devComm, flagcxDevMem mem,
                                  int myRank, int nRanks, int valueBase) {
  // Each rank writes to ALL peers: peer's buf[myRank] = valueBase + myRank
  for (int peer = 0; peer < nRanks; peer++) {
    size_t offset = (size_t)myRank * sizeof(int);
    int *dst = (int *)flagcxGetPeerPointer(mem, offset, flagcxTeamIntra(devComm), peer);
    if (dst) {
      *dst = valueBase + myRank;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel: test flagcxGetPeerPointer without team argument
// ---------------------------------------------------------------------------
__global__ void kernelWriteToPeerNoTeam(flagcxDevComm devComm, flagcxDevMem mem,
                                        int myRank, int nRanks, int valueBase) {
  for (int peer = 0; peer < nRanks; peer++) {
    size_t offset = (size_t)myRank * sizeof(int);
    int *dst = (int *)flagcxGetPeerPointer(mem, offset, peer);
    if (dst) {
      *dst = valueBase + myRank;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel: test flagcxSymPtr<int>::peerPtr
// ---------------------------------------------------------------------------
__global__ void kernelSymPtrPeerPtr(flagcxDevComm devComm, flagcxDevMem mem,
                                    int myRank, int nRanks, int valueBase) {
  for (int peer = 0; peer < nRanks; peer++) {
    flagcxSymPtr<int> symPtr(mem, (size_t)myRank * sizeof(int));
    int *dst = symPtr.peerPtr(flagcxTeamIntra(devComm), peer);
    if (dst) {
      *dst = valueBase + myRank;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel: verify getIntraPointer correctness
// Prints peerPtrs entries, checks self-pointer consistency, then writes
// using getIntraPointer directly.
// ---------------------------------------------------------------------------
__global__ void kernelVerifyIntraPointer(flagcxDevComm devComm, flagcxDevMem mem,
                                         int myRank, int nRanks) {
  // Print peerPtrs table
  printf("[VERIFY] rank%d: rawPtr=%p peerPtrs=%p intraRank=%d nRanks=%d\n",
         myRank, mem._winBase.rawPtr, (void*)mem._winBase.peerPtrs,
         mem._winBase.intraRank, nRanks);

  if (mem._winBase.peerPtrs) {
    for (int i = 0; i < nRanks; i++) {
      printf("[VERIFY] rank%d: peerPtrs[%d] = %p\n",
             myRank, i, mem._winBase.peerPtrs[i]);
    }
    // Self-pointer check: peerPtrs[myRank] should == rawPtr
    void *selfPtr = mem._winBase.peerPtrs[myRank];
    if (selfPtr == mem._winBase.rawPtr) {
      printf("[VERIFY] rank%d: SELF-CHECK OK (peerPtrs[%d] == rawPtr)\n",
             myRank, myRank);
    } else {
      printf("[VERIFY] rank%d: SELF-CHECK FAIL (peerPtrs[%d]=%p != rawPtr=%p)\n",
             myRank, myRank, selfPtr, mem._winBase.rawPtr);
    }
  } else {
    printf("[VERIFY] rank%d: peerPtrs is NULL!\n", myRank);
  }

  // Write using flagcxGetIntraPointer (the function under test)
  for (int peer = 0; peer < nRanks; peer++) {
    size_t offset = (size_t)myRank * sizeof(int);
    int *dst = (int *)flagcxGetIntraPointer(mem, offset, peer);
    printf("[VERIFY] rank%d: getIntraPointer(offset=%lu, peer=%d) = %p\n",
           myRank, (unsigned long)offset, peer, (void*)dst);
    if (dst) {
      *dst = myRank;
      __threadfence_system();
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel: offset correctness — diagonal write
// rank r writes value (r*1000+peer) at peer's buf[r*nRanks+r]
// ---------------------------------------------------------------------------
__global__ void kernelOffsetTest(flagcxDevComm devComm, flagcxDevMem mem,
                                 int myRank, int nRanks) {
  for (int peer = 0; peer < nRanks; peer++) {
    size_t offset = (size_t)(myRank * nRanks + myRank) * sizeof(int);
    int *dst = (int *)flagcxGetPeerPointer(mem, offset, flagcxTeamIntra(devComm), peer);
    if (dst) {
      *dst = myRank * 1000 + peer;
    }
  }
}

// ---------------------------------------------------------------------------
// Helper: run a test
// ---------------------------------------------------------------------------
bool runTest(const char *name, int testNum,
             flagcxDeviceHandle_t devHandle, flagcxDevComm_t devComm,
             flagcxDevMem_t devMem, void *regBuff, size_t bufSize,
             flagcxStream_t stream, int *hostBuf,
             int proc, int totalProcs, int color,
             void (*verifyFn)(int *hostBuf, int proc, int totalProcs, bool &pass)) {
  // Clear buffer
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 0, bufSize, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Launch kernel based on test number
  if (testNum == 1) {
    kernelWriteToPeer<<<1, 1, 0, stream->base>>>(
        flagcxDevComm(*devComm), flagcxDevMem(*devMem), proc, totalProcs, 0);
  } else if (testNum == 2) {
    kernelWriteToPeerNoTeam<<<1, 1, 0, stream->base>>>(
        flagcxDevComm(*devComm), flagcxDevMem(*devMem), proc, totalProcs, 100);
  } else if (testNum == 3) {
    kernelSymPtrPeerPtr<<<1, 1, 0, stream->base>>>(
        flagcxDevComm(*devComm), flagcxDevMem(*devMem), proc, totalProcs, 200);
  } else if (testNum == 4) {
    kernelOffsetTest<<<1, 1, 0, stream->base>>>(
        flagcxDevComm(*devComm), flagcxDevMem(*devMem), proc, totalProcs);
  }

  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Copy back
  FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, regBuff, bufSize,
                                      flagcxMemcpyDeviceToHost, NULL));

  bool pass = true;
  verifyFn(hostBuf, proc, totalProcs, pass);

  int ok = pass ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (proc == 0 && color == 0)
    printf("  %-50s %s\n", name, ok ? "PASS" : "FAIL");
  return ok == 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  parser args(argc, argv);
  int localRegister = args.getLocalRegister();

  flagcxHandlerGroup_t handler;
  FLAGCXCHECK(flagcxHandleInit(&handler));
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = args.getSplitMask();
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  // Buffer: nRanks * nRanks ints (enough for all tests)
  size_t bufSize = (size_t)totalProcs * totalProcs * sizeof(int);

  void *regBuff = nullptr;
  void *regHandle = nullptr;
  flagcxWindow_t win = nullptr;
  flagcxDevMem_t devMem = nullptr;

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize));
    FLAGCXCHECK(flagcxCommWindowRegister(comm, regBuff, bufSize, &win,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize));
    FLAGCXCHECK(flagcxCommRegister(comm, regBuff, bufSize, &regHandle));
  } else {
    FLAGCXCHECK(
        devHandle->deviceMalloc(&regBuff, bufSize, flagcxMemDevice, NULL));
  }

  // Create device communicator (no barrier needed for this test)
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create device memory handle
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  int *hostBuf = (int *)malloc(bufSize);
  int totalPass = 0, totalTests = 4;

  if (proc == 0 && color == 0) {
    printf("\n# flagcxGetPeerPointer Correctness Test\n");
    printf("# nRanks=%d, regMode=%s\n\n", totalProcs,
           localRegister == 2   ? "window"
           : localRegister == 1 ? "ipc"
                                : "raw");
  }

  // T1: flagcxGetPeerPointer(mem, offset, team, peer)
  {
    auto verify = [](int *buf, int proc, int nRanks, bool &pass) {
      for (int i = 0; i < nRanks; i++) {
        if (buf[i] != i) {
          printf("  rank%d: T1 FAIL [%d]: got %d expected %d\n", proc, i, buf[i], i);
          pass = false;
        }
      }
    };
    if (runTest("T1: GetPeerPointer(mem, off, team, peer)", 1,
                devHandle, devComm, devMem, regBuff, bufSize, stream, hostBuf,
                proc, totalProcs, color, verify))
      totalPass++;
  }

  // T2: flagcxGetPeerPointer(mem, offset, peer) — no team
  {
    auto verify = [](int *buf, int proc, int nRanks, bool &pass) {
      for (int i = 0; i < nRanks; i++) {
        if (buf[i] != i + 100) {
          printf("  rank%d: T2 FAIL [%d]: got %d expected %d\n", proc, i, buf[i], i + 100);
          pass = false;
        }
      }
    };
    if (runTest("T2: GetPeerPointer(mem, off, peer) no-team", 2,
                devHandle, devComm, devMem, regBuff, bufSize, stream, hostBuf,
                proc, totalProcs, color, verify))
      totalPass++;
  }

  // T3: flagcxSymPtr<int>::peerPtr(team, peer)
  {
    auto verify = [](int *buf, int proc, int nRanks, bool &pass) {
      for (int i = 0; i < nRanks; i++) {
        if (buf[i] != i + 200) {
          printf("  rank%d: T3 FAIL [%d]: got %d expected %d\n", proc, i, buf[i], i + 200);
          pass = false;
        }
      }
    };
    if (runTest("T3: SymPtr<int>::peerPtr(team, peer)", 3,
                devHandle, devComm, devMem, regBuff, bufSize, stream, hostBuf,
                proc, totalProcs, color, verify))
      totalPass++;
  }

  // T4: Offset correctness (diagonal write)
  {
    auto verify = [](int *buf, int proc, int nRanks, bool &pass) {
      for (int r = 0; r < nRanks; r++) {
        int idx = r * nRanks + r;
        int expected = r * 1000 + proc;
        if (buf[idx] != expected) {
          printf("  rank%d: T4 FAIL [%d]: got %d expected %d\n", proc, idx, buf[idx], expected);
          pass = false;
        }
      }
    };
    if (runTest("T4: Offset correctness (diagonal write)", 4,
                devHandle, devComm, devMem, regBuff, bufSize, stream, hostBuf,
                proc, totalProcs, color, verify))
      totalPass++;
  }

  if (proc == 0 && color == 0) {
    printf("\n# Result: %d/%d tests passed\n\n", totalPass, totalTests);
  }

  // Cleanup — follow test_internode_twosided.cpp order
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, devMem));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, win));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommDeregister(comm, regHandle));
  }

  FLAGCXCHECK(flagcxCommDestroy(comm));

  if (localRegister >= 1) {
    FLAGCXCHECK(flagcxMemFree(regBuff));
  } else {
    FLAGCXCHECK(devHandle->deviceFree(regBuff, flagcxMemDevice, NULL));
  }
  free(hostBuf);
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}