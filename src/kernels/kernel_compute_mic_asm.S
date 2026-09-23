    .text
    .globl sfbench_k1om_fma_loop
    .type sfbench_k1om_fma_loop,@function
sfbench_k1om_fma_loop:
    test %rdi,%rdi
    je .Lsfbench_k1om_done

    vbroadcastsd .Lsfbench_one(%rip), %zmm0
    vbroadcastsd .Lsfbench_one(%rip), %zmm1
    vbroadcastsd .Lsfbench_one(%rip), %zmm2
    vbroadcastsd .Lsfbench_one(%rip), %zmm3
    vbroadcastsd .Lsfbench_one(%rip), %zmm4
    vbroadcastsd .Lsfbench_one(%rip), %zmm5
    vbroadcastsd .Lsfbench_one(%rip), %zmm6
    vbroadcastsd .Lsfbench_one(%rip), %zmm7
    vbroadcastsd .Lsfbench_one(%rip), %zmm8
    vbroadcastsd .Lsfbench_one(%rip), %zmm9
    vbroadcastsd .Lsfbench_one(%rip), %zmm10
    vbroadcastsd .Lsfbench_one(%rip), %zmm11
    vbroadcastsd .Lsfbench_one(%rip), %zmm12
    vbroadcastsd .Lsfbench_one(%rip), %zmm13
    vbroadcastsd .Lsfbench_mul(%rip), %zmm14
    vbroadcastsd .Lsfbench_add(%rip), %zmm15

.Lsfbench_k1om_loop:
    vfmadd231pd %zmm14, %zmm15, %zmm0
    vfmadd231pd %zmm14, %zmm15, %zmm1
    vfmadd231pd %zmm14, %zmm15, %zmm2
    vfmadd231pd %zmm14, %zmm15, %zmm3
    vfmadd231pd %zmm14, %zmm15, %zmm4
    vfmadd231pd %zmm14, %zmm15, %zmm5
    vfmadd231pd %zmm14, %zmm15, %zmm6
    vfmadd231pd %zmm14, %zmm15, %zmm7
    vfmadd231pd %zmm14, %zmm15, %zmm8
    vfmadd231pd %zmm14, %zmm15, %zmm9
    vfmadd231pd %zmm14, %zmm15, %zmm10
    vfmadd231pd %zmm14, %zmm15, %zmm11
    vfmadd231pd %zmm14, %zmm15, %zmm12
    vfmadd231pd %zmm14, %zmm15, %zmm13
    dec %rdi
    jne .Lsfbench_k1om_loop

.Lsfbench_k1om_done:
    ret

    .macro SFBENCH_K1OM_MASKED_FP64_STEP
    vfmadd231pd %zmm14, %zmm15, %zmm0{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm1{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm2{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm3{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm4{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm5{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm6{%k1}
    vfmadd231pd %zmm14, %zmm15, %zmm7{%k1}
    .endm

    .globl sfbench_k1om_fma_loop_masked_fp64
    .type sfbench_k1om_fma_loop_masked_fp64,@function
sfbench_k1om_fma_loop_masked_fp64:
    test %rdi,%rdi
    je .Lsfbench_k1om_masked_done

    mov %rdi,%rcx
    shr $3,%rdi
    and $7,%rcx

    mov $1,%eax
    kmov %eax,%k1
    vbroadcastsd .Lsfbench_one(%rip), %zmm0
    vbroadcastsd .Lsfbench_one(%rip), %zmm1
    vbroadcastsd .Lsfbench_one(%rip), %zmm2
    vbroadcastsd .Lsfbench_one(%rip), %zmm3
    vbroadcastsd .Lsfbench_one(%rip), %zmm4
    vbroadcastsd .Lsfbench_one(%rip), %zmm5
    vbroadcastsd .Lsfbench_one(%rip), %zmm6
    vbroadcastsd .Lsfbench_one(%rip), %zmm7
    vbroadcastsd .Lsfbench_mul(%rip), %zmm14
    vbroadcastsd .Lsfbench_add(%rip), %zmm15

.Lsfbench_k1om_masked_loop:
    test %rdi,%rdi
    je .Lsfbench_k1om_masked_tail
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    SFBENCH_K1OM_MASKED_FP64_STEP
    dec %rdi
    jne .Lsfbench_k1om_masked_loop

.Lsfbench_k1om_masked_tail:
    test %rcx,%rcx
    je .Lsfbench_k1om_masked_done
.Lsfbench_k1om_masked_tail_loop:
    SFBENCH_K1OM_MASKED_FP64_STEP
    dec %rcx
    jne .Lsfbench_k1om_masked_tail_loop

.Lsfbench_k1om_masked_done:
    ret

    .globl sfbench_k1om_fma_loop_fp32
    .type sfbench_k1om_fma_loop_fp32,@function
sfbench_k1om_fma_loop_fp32:
    test %rdi,%rdi
    je .Lsfbench_k1om_fp32_done

    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm0
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm1
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm2
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm3
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm4
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm5
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm6
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm7
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm8
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm9
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm10
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm11
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm12
    vbroadcastss .Lsfbench_one_fp32(%rip), %zmm13
    vbroadcastss .Lsfbench_mul_fp32(%rip), %zmm14
    vbroadcastss .Lsfbench_add_fp32(%rip), %zmm15

.Lsfbench_k1om_fp32_loop:
    vfmadd231ps %zmm14, %zmm15, %zmm0
    vfmadd231ps %zmm14, %zmm15, %zmm1
    vfmadd231ps %zmm14, %zmm15, %zmm2
    vfmadd231ps %zmm14, %zmm15, %zmm3
    vfmadd231ps %zmm14, %zmm15, %zmm4
    vfmadd231ps %zmm14, %zmm15, %zmm5
    vfmadd231ps %zmm14, %zmm15, %zmm6
    vfmadd231ps %zmm14, %zmm15, %zmm7
    vfmadd231ps %zmm14, %zmm15, %zmm8
    vfmadd231ps %zmm14, %zmm15, %zmm9
    vfmadd231ps %zmm14, %zmm15, %zmm10
    vfmadd231ps %zmm14, %zmm15, %zmm11
    vfmadd231ps %zmm14, %zmm15, %zmm12
    vfmadd231ps %zmm14, %zmm15, %zmm13
    dec %rdi
    jne .Lsfbench_k1om_fp32_loop

.Lsfbench_k1om_fp32_done:
    ret

    .section .rodata
    .align 64
.Lsfbench_one:
    .double 1.0
.Lsfbench_mul:
    .double 1.0000000001
.Lsfbench_add:
    .double 0.0000000001
.Lsfbench_one_fp32:
    .float 1.0
.Lsfbench_mul_fp32:
    .float 1.0000001
.Lsfbench_add_fp32:
    .float 0.0000001
