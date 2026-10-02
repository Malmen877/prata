//! Keep onnxruntime's int8 kernels off Intel AMX (Linux x86_64 only).
//!
//! On a Xeon with AMX running as a VM guest, the int8 encoder gave different and
//! sometimes garbage output (41 % WER instead of 5.4 % on a test clip) whenever
//! the process was preempted: the AMX tile state was not preserved. Without AMX,
//! MLAS uses its AVX-512 VNNI / AVX2 kernels; output was then identical in every
//! run, under load or not, and only about 10 % slower.
//!
//! MLAS enables AMX by asking the kernel for permission with
//! `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`. A seccomp filter makes
//! exactly that call fail with EPERM (everything else is allowed), so MLAS falls
//! back by itself. The filter is installed for every thread of the process
//! (SECCOMP_FILTER_FLAG_TSYNC) before any onnxruntime session exists, and is
//! inherited by threads created later (the onnxruntime thread pool). It only
//! affects this prata process (and children), which is a one-shot CLI run.
//! `PRATA_SNABB_AMX=1` keeps AMX enabled.

const ARCH_REQ_XCOMP_PERM: u32 = 0x1023;
const AUDIT_ARCH_X86_64: u32 = 0xC000_003E;

pub fn disable_amx() {
    if std::env::var("PRATA_SNABB_AMX").map(|v| v == "1").unwrap_or(false) {
        return;
    }
    use libc::{sock_filter, sock_fprog};
    const LD_W_ABS: u16 = (libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16;
    const JEQ_K: u16 = (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16;
    const RET_K: u16 = (libc::BPF_RET | libc::BPF_K) as u16;
    let f = |code: u16, jt: u8, jf: u8, k: u32| sock_filter { code, jt, jf, k };
    // struct seccomp_data { int nr; u32 arch; u64 ip; u64 args[6]; }
    let prog = [
        f(LD_W_ABS, 0, 0, 4),                                  // arch
        f(JEQ_K, 0, 5, AUDIT_ARCH_X86_64),                     // not x86_64 -> allow
        f(LD_W_ABS, 0, 0, 0),                                  // nr
        f(JEQ_K, 0, 3, libc::SYS_arch_prctl as u32),           // not arch_prctl -> allow
        f(LD_W_ABS, 0, 0, 16),                                 // args[0] (low 32 bits)
        f(JEQ_K, 0, 1, ARCH_REQ_XCOMP_PERM),                   // other option -> allow
        f(RET_K, 0, 0, libc::SECCOMP_RET_ERRNO | libc::EPERM as u32),
        f(RET_K, 0, 0, libc::SECCOMP_RET_ALLOW),
    ];
    let fprog = sock_fprog { len: prog.len() as u16, filter: prog.as_ptr() as *mut sock_filter };
    // SAFETY: plain prctl calls with a valid, live filter program.
    let ok = unsafe {
        libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0
            && (libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER as libc::c_long,
                libc::SECCOMP_FILTER_FLAG_TSYNC as libc::c_long,
                &fprog as *const sock_fprog,
            ) == 0
                || libc::prctl(libc::PR_SET_SECCOMP, libc::SECCOMP_MODE_FILTER, &fprog as *const sock_fprog) == 0)
    };
    if !ok {
        eprintln!("[warn] snabb: could not restrict AMX (seccomp unavailable here); continuing with onnxruntime defaults");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn amx_permission_request_is_refused_in_a_child() {
        // run in a forked child so the filter does not affect the test runner
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0);
            if pid == 0 {
                super::disable_amx();
                let r = libc::syscall(libc::SYS_arch_prctl, super::ARCH_REQ_XCOMP_PERM as libc::c_long, 18 as libc::c_long);
                let e = *libc::__errno_location();
                let other = libc::syscall(libc::SYS_getpid);
                libc::_exit(if r == -1 && e == libc::EPERM && other > 0 { 0 } else { 1 });
            }
            let mut status = 0;
            libc::waitpid(pid, &mut status, 0);
            assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0, "status {status}");
        }
    }
}
