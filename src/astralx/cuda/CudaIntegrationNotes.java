package astralx.cuda;

public final class CudaIntegrationNotes {
    private CudaIntegrationNotes() {}

    public static String note() {
        return "CUDA kernels are in src/cuda/. Integrate via JNI or JavaCPP for GPU hash/bin lookups and batched scoring.";
    }
}
