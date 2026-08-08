// Correctness test for hinge-loss: output[0] = mean over n of max(0, 1 - pred*tgt).
// Scalar loss (output has size 1), mean reduction. If your problem uses a sum
// reduction instead, adjust the /n below.

#include "test_utils.cuh"
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n);

int main() {
    test::seed();
    size_t n = 1024;

    float* h_p = new float[n];
    float* h_t = new float[n];
    float h_o = 0.0f, h_ref = 0.0f;
    test::fill_random(h_p, n);
    test::fill_random(h_t, n);

    test::DBuf<float> d_p(n), d_t(n), d_o(1);   // d_o is zero-initialized by DBuf
    d_p.upload(h_p);
    d_t.upload(h_t);

    solution(d_p, d_t, d_o, n);
    test::check_cuda("hinge-loss");
    d_o.download(&h_o);

    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = 1.0 - static_cast<double>(h_p[i]) * h_t[i];
        acc += e > 0.0 ? e : 0.0;
    }
    h_ref = static_cast<float>(acc / n);

    int bad = test::compare("hinge-loss", &h_o, &h_ref, 1, 1e-3f, 1e-4f);
    int rc = test::report("hinge-loss", bad, 1);
    delete[] h_p;
    delete[] h_t;
    return rc;
}
