subroutine f(mu, Sigma, H, R, data, mu_2, Sigma_2)
  implicit none

  real(kind=8), intent(in) :: mu(:)
  real(kind=8), intent(in) :: Sigma(:,:)
  real(kind=8), intent(in) :: H(:,:)
  real(kind=8) :: R(:,:)
  real(kind=8) :: data(:)
  real(kind=8), intent(out) :: mu_2(:)
  real(kind=8), intent(out) :: Sigma_2(:,:)

! Variable Declarations !   -- Cut for space --

  call dcopy(n**2, Sigma, 1, Sigma_2, 1)
  call dgemm('N', 'N', k, 1, n, 1.0d+0, H, k, mu, n, -1.0d+0, data, k)
  call dcopy(n, mu, 1, mu_2, 1)
  call dcopy(k*n, H, 1, H_2, 1)
  call dsymm('R', 'U', k, n, 1.0d+0, Sigma, n, H, k, 0.0d+0, var_17, k)
  call dgemm('N', 'T', k, k, n, 1.0d+0, H, k, var_17, k, 1.0d+0, R, k)
  call dposv('U', k, n, R, k, H_2, k, INFO)
  call dpotrs('U', k, 1, R, k, data, k, INFO)
  call dgemm('T', 'N', n, 1, k, 1.0d+0, H, k, data, k, 0.0d+0, var_12, n)
  call dsymm('L', 'U', n, 1, 1.0d+0, Sigma, n, var_12, n, 1.0d+0, mu_2, n)
  call dgemm('T', 'N', n, n, k, 1.0d+0, H, k, H_2, k, 0.0d+0, var_19, n)
  call dsymm('R', 'U', n, n, 1.0d+0, Sigma, n, var_19, n, 0.0d+0, var_18, n)
  call dsymm('L', 'U', n, n, -1.0d+0, Sigma_2, n, var_18, n, 1.0d+0, Sigma_2, n)
  return
end subroutine f
