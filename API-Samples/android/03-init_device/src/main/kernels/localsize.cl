/*
clspv localsize.cl -o ../assets/clspv_tests/localsize.spv -descriptormap ../assets/clspv_tests/localsize.spvmap
*/

typedef struct {
	int	x;
	int	y;
	int z;
} S;

__kernel void ReadLocalSize(__global S* s)
{
	s->x = get_local_size(0);
	s->y = get_local_size(1);
	s->z = get_local_size(2);
}
