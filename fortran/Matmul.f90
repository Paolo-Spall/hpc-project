program Matmul

  use mpi

  implicit none

  character(LEN=50) :: stringa

  integer, parameter :: N = 8
  integer :: nprocs, ierr, ecode, proc_me, status(MPI_STATUS_SIZE)
  integer :: count, i, j, iglob, k, z, rem
  real, allocatable, dimension(:,:) :: A, B, C, buffer
  real, allocatable, dimension(:) :: sup
  integer :: Nrow, Ncol

  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nprocs,ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,proc_me,ierr)
  !$ Get local matrix dimensions
  Nrow = N / nprocs
  NCol = N

!!!
!!! STEP 1: ALLOCATE WORKSPACE
!!!

  allocate(A(Nrow,Ncol))
  allocate(B(Nrow,Ncol))
  allocate(C(Nrow,Ncol))
  allocate(buffer(Nrow,Ncol))
  allocate(sup(Ncol))

  !$ Avoid the program to run with a number of processes that does not divide exactly dim
  rem = mod(N,nprocs)
  if ( rem .ne. 0 ) then
     if (proc_me .eq. 0) print *,"The number of process must divide ",N," exactly."
     call MPI_Abort(MPI_COMM_WORLD,ecode,ierr)
  end if


!!!
!!! STEP 2:Initialize matrices
!!!
  do j=1,Ncol
     do i=1,Nrow
        A(i,j) = ((Nrow * proc_me)+i) * j
        B(i,j) = 1.0 / A(i,j)
     enddo
  enddo
  C = 0.0

!!!
!!! STEP3: MPI_ALLGATHER and MATRIX MATRIX MULTIPLICATIONS
!!! 
  do count=0,nprocs-1
     call MPI_Allgather(B(:,(count*Nrow)+1:count*Nrow+Nrow), Nrow*Nrow, MPI_REAL, buffer, Nrow*Nrow, MPI_REAL, MPI_COMM_WORLD, ierr)
     if (proc_me .eq. 0) then
        print *
        print *, buffer(1,:)
        print *, buffer(2,:)
     end if
     do k = (count*Nrow) + 1, (count+1) * Nrow
        do j = 1, Ncol
           do i = 1, Nrow
              C(i,k)  = C(i,k) + A(i,j) * buffer(mod(j-1,NRow)+1,((j-1)/Nrow)*Nrow + k - (count*Nrow))
           end do
        end do
    end do
  end do

  write(stringa, *) Ncol

!!
!! Print matrices
!!

  if (proc_me == 0) then
     print *
     do i=1,Nrow
        print '('//trim(stringa)//'(F8.2))', A(i,:)
     enddo

     do k=1, nprocs-1
        call MPI_RECV(A, Nrow*Ncol, MPI_REAL, k, 0, MPI_COMM_WORLD, status, ierr)
        do i=1,Nrow
           print '('//trim(stringa)//'(F8.2))', A(i,:)
        enddo
     enddo
  else
     call MPI_SEND(A, Nrow*Ncol, MPI_REAL, 0, 0, MPI_COMM_WORLD, ierr)
  endif

  if (proc_me == 0) then
     print *
     do i=1,Nrow
        print '('//trim(stringa)//'(F8.2))', B(i,:)
     enddo

     do k=1, nprocs-1
        call MPI_RECV(B, Nrow*Ncol, MPI_REAL, k, 0, MPI_COMM_WORLD, status, ierr)
        do i=1,Nrow
           print '('//trim(stringa)//'(F8.2))', B(i,:)
        enddo
     enddo
  else
     call MPI_SEND(B, Nrow*Ncol, MPI_REAL, 0, 0, MPI_COMM_WORLD, ierr)
  endif

  if (proc_me == 0) then
     print *
     do i=1,Nrow
        print '('//trim(stringa)//'(F8.2))', C(i,:)
     enddo

     do k=1, nprocs-1
        call MPI_RECV(C, Nrow*Ncol, MPI_REAL, k, 0, MPI_COMM_WORLD, status, ierr)
        do i=1,Nrow
           print '('//trim(stringa)//'(F8.2))', C(i,:)
        enddo
     enddo
  else
     call MPI_SEND(C, Nrow*Ncol, MPI_REAL, 0, 0, MPI_COMM_WORLD, ierr)
  endif

  !$ Free workspace
  deallocate(A,B,C,buffer,sup)
  call MPI_FINALIZE(ierr)

end program Matmul
