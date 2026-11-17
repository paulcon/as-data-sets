"""Patch TurbSim's Coh2H (OpenFAST v3.5.5, modules/turbsim/src/TSsubs.f90) as it was for these runs.

    python patch_turbsim.py <openfast source>/modules/turbsim/src/TSsubs.f90

If the GENERAL-model coherence matrix of a component is not positive definite at a frequency, the
Cholesky factorization is retried with the off-diagonal coherences divided by (1 + eps), eps = 1e-5,
1e-4, ..., 10 (the diagonal stays 1, so the point variances are unchanged). Each retry that succeeds
prints 'ASD_NUGGET <component> <frequency index> <eps>'; openfast_driver.py counts these lines
(nugget_n) and keeps the largest eps (nugget_max_eps) for each run. Unpatched TurbSim stops with an
error at such a frequency."""
import sys
p = sys.argv[1]; t = open(p).read()
old = "   CALL LAPACK_pptrf( 'L', NPts, TRH(Indx:), ErrStat, ErrMsg )"
assert t.count(old) == 1, t.count(old)
i = t.index(old); j = t.index('\n', i)
new = """   BLOCK
      REAL(ReKi), ALLOCATABLE :: TRH0(:)
      REAL(ReKi)              :: eps
      INTEGER                 :: K, II, JJ, NP
      NP = NPts*(NPts+1)/2
      ALLOCATE(TRH0(NP))
      TRH0 = TRH(Indx:Indx+NP-1)
      eps = 0.0_ReKi
      DO K = 0, 7
         IF (K > 0) THEN
            eps = 1.0E-5_ReKi * 10.0_ReKi**(K-1)
            II = Indx
            DO JJ = 1, NPts
               TRH(II) = TRH0(II-Indx+1)
               IF (JJ < NPts) TRH(II+1:II+NPts-JJ) = TRH0(II-Indx+2:II-Indx+1+NPts-JJ) / (1.0_ReKi + eps)
               II = II + NPts - JJ + 1
            END DO
         END IF
         CALL LAPACK_pptrf( 'L', NPts, TRH(Indx:), ErrStat, ErrMsg )
         IF (ErrStat < AbortErrLev) EXIT
      END DO
      IF (eps > 0.0_ReKi .AND. ErrStat < AbortErrLev) CALL WrScr('ASD_NUGGET '//Comp(IVec)//' '//TRIM(Int2LStr(IFreq))//' '//TRIM(Num2LStr(eps)))
      DEALLOCATE(TRH0)
   END BLOCK"""
t = t[:i] + new + t[j:]
open(p, 'w').write(t); print('patched', p)
