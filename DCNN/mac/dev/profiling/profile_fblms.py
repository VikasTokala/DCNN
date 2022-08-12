import mac
import mac.test.test_est as test

if __name__ == '__main__':

    N_rtf_t = 1024

    ir_type = 'nuance'

    sm = mac.sm.TestSignalModel(ir_type=ir_type, Ts=0.5, len_w=1024, overlap=None, SNR_db=6)

    rtf_fblms, y_hat_fblms, mse_fblms = test.test_fblms_rtf(sm)