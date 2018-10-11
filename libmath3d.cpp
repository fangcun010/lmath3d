extern "C"{
#include <lauxlib.h>
}
#include <math.h>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.inl>

//#define						EPS						1e-9

int isGLM=0;

struct vector{
	float x,y,z,w;
};

struct matrix{
	float mat[4][4];
};

struct quaternion{
	float x,y,z,w;
};

void
float_eps(float *val){
#ifdef EPS
	if(fabs(*val)<EPS) *val=0.0f;
#endif
}

void
vec_eps(struct vector *v){
#ifdef EPS
	if(fabs(v->x)<EPS) v->x=0.0f;
	if(fabs(v->y)<EPS) v->y=0.0f;
	if(fabs(v->z)<EPS) v->z=0.0f;
	if(fabs(v->w-1.0f)<EPS) v->w=1.0f;
#endif
}

void
assign_mat(glm::mat4x4 &mat,struct matrix *m){
	for(int i=0;i<4;i++)
	for(int j=0;j<4;j++)
		mat[i][j]=m->mat[i][j];
}

void assign_vec3(glm::vec3 &vec,struct vector *v){
	vec.x=v->x;vec.y=v->y;vec.z=v->z;
}

void rassign_vec3(struct vector *v,glm::vec3 &vec){
	v->x=vec.x;v->y=vec.y;v->z=vec.z;
}

void
rassign_mat(struct matrix *m,glm::mat4x4 &mat){
	for(int i=0;i<4;i++)
	for(int j=0;j<4;j++)
		m->mat[i][j]=mat[i][j];
}

void
assign_vec(glm::vec4 &vec,struct vector *v){
	vec.x=v->x;vec.y=v->y;vec.z=v->z;vec.w=v->w;
}

void
rassign_vec(struct vector *v,glm::vec4 &vec){
	v->x=vec.x;v->y=vec.y;v->z=vec.z;v->w=vec.w;
}

void
vec_add(struct vector *vresult,struct vector *v1,struct vector *v2){
	vresult->x=v1->x+v2->x;
	vresult->y=v1->y+v2->y;
	vresult->z=v1->z+v2->z;
	vresult->w=1.0f;
	vec_eps(vresult);
}

void
vec_sub(struct vector *vresult, struct vector *v1,struct vector *v2){
	vresult->x=v1->x-v2->x;
	vresult->y=v1->y-v2->y;
	vresult->z=v1->z-v2->z;
	vresult->w=1.0f;
	vec_eps(vresult);
}

void
vec_dot(float *res,struct vector *v1,struct vector *v2){
	*res=v1->x*v2->x+v1->y*v2->y+v1->z*v2->z;
	float_eps(res);
}

void
vec_cross(struct vector *vresult,struct vector *v1,struct vector *v2){
	vresult->x=(v1->y*v2->z-v1->z*v2->y);
	vresult->y=-(v1->x*v2->z-v1->z*v2->x);
	vresult->z=(v1->x*v2->y-v1->y*v2->x);
	vresult->w=1.0f;
	vec_eps(vresult);
}

void
vec_normalize(struct vector *vresult,struct vector *v){
	float len=sqrtf(v->x*v->x+v->y*v->y+v->z*v->z);
	vresult->x=v->x/len;
	vresult->y=v->y/len;
	vresult->z=v->z/len;
	vresult->w=1.0f;
	vec_eps(vresult);
}

void
mat_identity(struct matrix *m){
	m->mat[0][0]=1;		m->mat[1][0]=0;		m->mat[2][0]=0;		m->mat[3][0]=0;
	m->mat[0][1]=0;		m->mat[1][1]=1;		m->mat[2][1]=0;		m->mat[3][1]=0;
	m->mat[0][2]=0;		m->mat[1][2]=0;		m->mat[2][2]=1;		m->mat[3][2]=0;
	m->mat[0][3]=0;		m->mat[1][3]=0;		m->mat[2][3]=0;		m->mat[3][3]=1;
}

void
mat_mul(struct matrix *result,struct matrix *m1,struct matrix *m2){
	struct matrix mat;
	for(int i=0;i<4;i++)
	for(int j=0;j<4;j++)
		mat.mat[i][j]=0.0f;

	for(int i=0;i<4;i++)
	for(int j=0;j<4;j++)
	for(int k=0;k<4;k++)
		mat.mat[j][i]+=m1->mat[k][i]*m2->mat[j][k];

	for(int i=0;i<4;i++)
	for(int j=0;j<4;j++)
		result->mat[i][j]=mat.mat[i][j];
}

void
mat_mul_vec(struct vector *vresult,struct matrix *m,struct vector *v){
	vresult->x=m->mat[0][0]*v->x+m->mat[1][0]*v->y+m->mat[2][0]*v->z+m->mat[3][0]*v->w;
	vresult->y=m->mat[0][1]*v->x+m->mat[1][1]*v->y+m->mat[2][1]*v->z+m->mat[3][1]*v->w;
	vresult->z=m->mat[0][2]*v->x+m->mat[1][2]*v->y+m->mat[2][2]*v->z+m->mat[3][2]*v->w;
	vresult->w=m->mat[0][3]*v->x+m->mat[1][3]*v->y+m->mat[2][3]*v->z+m->mat[3][3]*v->w;
	vresult->x/=vresult->w;
	vresult->y/=vresult->w;
	vresult->z/=vresult->w;
	vresult->w=1.0f;
	vec_eps(vresult);
}

void
mat_translate(struct matrix *m,float x,float y,float z){
	struct matrix mat;
	mat.mat[0][0]=1;	mat.mat[1][0]=0;	mat.mat[2][0]=0;	mat.mat[3][0]=x;
	mat.mat[0][1]=0;	mat.mat[1][1]=1;	mat.mat[2][1]=0;	mat.mat[3][1]=y;
	mat.mat[0][2]=0;	mat.mat[1][2]=0;	mat.mat[2][2]=1;	mat.mat[3][2]=z;
	mat.mat[0][3]=0;	mat.mat[1][3]=0;	mat.mat[2][3]=0;	mat.mat[3][3]=1;
	mat_mul(m,m,&mat);
}

void
mat_scale(struct matrix *m,float x,float y,float z){
	struct matrix mat;
	mat.mat[0][0]=x;	mat.mat[1][0]=0;	mat.mat[2][0]=0;	mat.mat[3][0]=0;
	mat.mat[0][1]=0;	mat.mat[1][1]=y;	mat.mat[2][1]=0;	mat.mat[3][1]=0;
	mat.mat[0][2]=0;	mat.mat[1][2]=0;	mat.mat[2][2]=z;	mat.mat[3][2]=0;
	mat.mat[0][3]=0;	mat.mat[1][3]=0;	mat.mat[2][3]=0;	mat.mat[3][3]=1;
	mat_mul(m,m,&mat);
}

void
mat_rotate_x(struct matrix *m,float r){
	struct matrix mat;
	mat.mat[0][0]=1;	mat.mat[1][0]=0;		mat.mat[2][0]=0;		mat.mat[3][0]=0;
	mat.mat[0][1]=0;	mat.mat[1][1]=cosf(r);	mat.mat[2][1]=-sinf(r);	mat.mat[3][1]=0;
	mat.mat[0][2]=0;	mat.mat[1][2]=sinf(r);	mat.mat[2][2]=cosf(r);	mat.mat[3][2]=0;
	mat.mat[0][3]=0;	mat.mat[1][3]=0;		mat.mat[2][3]=0;		mat.mat[3][3]=1;
	mat_mul(m,m,&mat);
}

void
mat_rotate_y(struct matrix *m,float r){
	struct matrix mat;
	mat.mat[0][0]=cosf(r);	mat.mat[1][0]=0;	mat.mat[2][0]=sinf(r);	mat.mat[3][0]=0;
	mat.mat[0][1]=0;		mat.mat[1][1]=1;	mat.mat[2][1]=0;		mat.mat[3][1]=0;
	mat.mat[0][2]=-sinf(r);	mat.mat[1][2]=0;	mat.mat[2][2]=cosf(r);	mat.mat[3][2]=0;
	mat.mat[0][3]=0;		mat.mat[1][3]=0;	mat.mat[2][3]=0;		mat.mat[3][3]=1;
	mat_mul(m,m,&mat);
}

void
mat_rotate_z(struct matrix *m,float r){
	struct matrix mat;
	mat.mat[0][0]=cosf(r);	mat.mat[1][0]=-sinf(r);	mat.mat[2][0]=0;	mat.mat[3][0]=0;
	mat.mat[0][1]=sinf(r);	mat.mat[1][1]=cosf(r);	mat.mat[2][1]=0;	mat.mat[3][1]=0;
	mat.mat[0][2]=0;		mat.mat[1][2]=0;		mat.mat[2][2]=1;	mat.mat[3][2]=0;
	mat.mat[0][3]=0;		mat.mat[1][3]=0;		mat.mat[2][3]=0;	mat.mat[3][3]=1;
	mat_mul(m,m,&mat);
}

void
mat_rotate(struct matrix *m,float r,float x,float y,float z){
	struct matrix mat;
	struct vector vec;
	vec.x=x;vec.y=y;vec.z=z;vec.w=1.0f;
	vec_normalize(&vec,&vec);

	mat.mat[0][0]=cosf(r)+(1.0f-cosf(r))*vec.x*vec.x;
	mat.mat[1][0]=(1.0f-cosf(r))*vec.x*vec.y-vec.z*sinf(r);
	mat.mat[2][0]=(1.0f-cosf(r))*vec.x*vec.z+vec.y*sinf(r);
	mat.mat[3][0]=0;

	mat.mat[0][1]=(1.0f-cosf(r))*vec.x*vec.y+vec.z*sinf(r);
	mat.mat[1][1]=cosf(r)+(1.0f-cosf(r))*vec.y*vec.y;
	mat.mat[2][1]=(1.0f-cosf(r))*vec.y*vec.z-vec.x*sinf(r);
	mat.mat[3][1]=0;

	mat.mat[0][2]=(1-cosf(r))*vec.x*vec.z-vec.y*sinf(r);
	mat.mat[1][2]=(1-cosf(r))*vec.y*vec.z+vec.x*sinf(r);
	mat.mat[2][2]=cosf(r)+(1.0f-cosf(r))*vec.z*vec.z;
	mat.mat[3][2]=0;

	mat.mat[0][3]=0;
	mat.mat[1][3]=0;
	mat.mat[2][3]=0;
	mat.mat[3][3]=1.0f;
	
	mat_mul(m,m,&mat);
}

void
mat_look_at(struct matrix *m,float eyeX,float eyeY,float eyeZ,
								float centerX,float centerY,float centerZ,
								float upX,float upY,float upZ){
	struct vector eye;
	struct vector center;
	struct vector up;
	struct vector r,u,v;
	struct vector t1,t2,t3;
	float v1,v2,v3;
	eye.x=eyeX;eye.y=eyeY;eye.z=eyeZ;
	center.x=centerX;center.y=centerY;center.z=centerZ;
	up.x=upX;up.y=upY;up.z=upZ;

	vec_sub(&v,&center,&eye);
	vec_normalize(&v,&v);
	vec_cross(&r,&v,&up);
	vec_normalize(&r,&r);
	vec_cross(&u,&r,&v);
	t3=eye;
	t1.x=-t3.x;t1.y=-t3.y;t1.z=-t3.z;
	t2=t1;
	vec_dot(&v1,&t1,&r);
	vec_dot(&v2,&t2,&u);
	vec_dot(&v3,&t3,&v);

	m->mat[0][0]=r.x;	m->mat[1][0]=r.y;		m->mat[2][0]=r.z;	m->mat[3][0]=v1;
	m->mat[0][1]=u.x;	m->mat[1][1]=u.y;		m->mat[2][1]=u.z;	m->mat[3][1]=v2;
	m->mat[0][2]=-v.x;	m->mat[1][2]=-v.y;		m->mat[2][2]=-v.z;	m->mat[3][2]=v3;
	m->mat[0][3]=0.0f;	m->mat[1][3]=0.0f;		m->mat[2][3]=0.0f;	m->mat[3][3]=1.0f;
}
/*void
mat_look_at(struct matrix *m,float eyeX,float eyeY,float eyeZ,
								float centerX,float centerY,float centerZ,
								float upX,float upY,float upZ){
	struct vector eye;
	struct vector center;
	struct vector up;
	struct vector r,u,v;
	struct vector t1,t2,t3;
	float v1,v2,v3;
	eye.x=eyeX;eye.y=eyeY;eye.z=eyeZ;
	center.x=centerX;center.y=centerY;center.z=centerY;
	up.x=upX;up.y=upY;up.z=upZ;

	vec_sub(&v,&eye,&center);
	vec_normalize(&v,&v);
	vec_cross(&r,&v,&up);
	vec_normalize(&r,&r);
	vec_cross(&u,&r,&v);
	t3=eye;
	t1.x=-t3.x;t1.y=-t3.y;t1.z=-t3.z;
	t2=t1;
	vec_dot(&v1,&t1,&r);
	vec_dot(&v2,&t2,&u);
	vec_dot(&v3,&t3,&v);

	m->mat[0][0]=r.x;	m->mat[1][0]=r.y;		m->mat[2][0]=r.z;	m->mat[3][0]=v1;
	m->mat[0][1]=u.x;	m->mat[1][1]=u.y;		m->mat[2][1]=u.z;	m->mat[3][1]=v2;
	m->mat[0][2]=v.x;	m->mat[1][2]=v.y;		m->mat[2][2]=v.z;	m->mat[3][2]=v3;
	m->mat[0][3]=0.0f;	m->mat[1][3]=0.0f;		m->mat[2][3]=0.0f;	m->mat[3][3]=1.0f;
}*/

void
mat_ortho(struct matrix *m,float left,float right,float bottom,float top,
				float near,float far){
	near=-near;far=-far;
	m->mat[0][0]=2.0f/(right-left);	m->mat[1][0]=0.0f;						m->mat[2][0]=0.0f;				m->mat[3][0]=-(right+left)/(right-left);
	m->mat[0][1]=0.0f;				m->mat[1][1]=2.0f/(top-bottom);			m->mat[2][1]=0.0f;				m->mat[3][1]=-(top+bottom)/(top-bottom);
	m->mat[0][2]=0.0f;				m->mat[1][2]=0.0;						m->mat[2][2]=2.0f/(far-near);	m->mat[3][2]=-(far+near)/(far-near);
	m->mat[0][3]=0.0f;				m->mat[1][3]=0.0f;						m->mat[2][3]=0.0f;				m->mat[3][3]=1.0f;
}

void
mat_frustum(struct matrix *m,float left,float right,float bottom,float top,
			float near,float far){
	m->mat[0][0]=(2.0f*near)/(right-left);	m->mat[1][0]=0.0f;						m->mat[2][0]=(right+left)/(right-left);	m->mat[3][0]=0.0f;
	m->mat[0][1]=0.0f;						m->mat[1][1]=(2.0f*near)/(top-bottom);	m->mat[2][1]=(top+bottom)/(top-bottom);	m->mat[3][1]=0.0f;
	m->mat[0][2]=0.0f;						m->mat[1][2]=0.0;						m->mat[2][2]=-(far+near)/(far-near);	m->mat[3][2]=-(2*far*near)/(far-near);
	m->mat[0][3]=0.0f;						m->mat[1][3]=0.0f;						m->mat[2][3]=-1.0f;						m->mat[3][3]=0.0f;
}

void
mat_perspective(struct matrix *m,float fovy,float aspect,float near, float far){
	fovy=fovy/180.0*M_PI;
	float c=1.0f/tanf(fovy/2.0f);
	m->mat[0][0]=c/aspect;	m->mat[1][0]=0.0f;		m->mat[2][0]=0.0f;						m->mat[3][0]=0.0f;
	m->mat[0][1]=0.0f;		m->mat[1][1]=c;			m->mat[2][1]=0.0f;						m->mat[3][1]=0.0f;
	m->mat[0][2]=0.0f;		m->mat[1][2]=0.0f;		m->mat[2][2]=-(far+near)/(far-near);	m->mat[3][2]=-(2*far*near)/(far-near);
	m->mat[0][3]=0.0f;		m->mat[1][3]=0.0f;		m->mat[2][3]=-1.0f;						m->mat[3][3]=0.0f;
}

static int
lmul_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==3){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		struct matrix *m1=(matrix *)lua_touserdata(L,2);
		struct matrix *m2=(matrix *)lua_touserdata(L,3);
		if(isGLM)
			mat_mul(m,m1,m2);
		else{
			glm::mat4x4 mat_m,mat_m1,mat_m2;
			assign_mat(mat_m,m);assign_mat(mat_m1,m1);assign_mat(mat_m2,m2);
			mat_m=mat_m1*mat_m2;
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lmul_vec(lua_State *L){
	int n=lua_gettop(L);
	if(n==3){
		struct vector *vresult=(vector *)lua_touserdata(L,1);
		struct matrix *m=(matrix *)lua_touserdata(L,2);
		struct vector *v=(vector *)lua_touserdata(L,3);
		struct vector v1=*v;
		if(isGLM)
			mat_mul_vec(vresult,m,&v1);
		else{
			glm::vec4 vec_vresult;
			glm::mat4x4 mat_m;
			glm::vec4 vec_v,vec_v1;
			assign_vec(vec_vresult,vresult);
			assign_mat(mat_m,m);
			assign_vec(vec_v,v);
			assign_vec(vec_v1,v);
			vec_vresult=mat_m*vec_v1;
			rassign_vec(vresult,vec_vresult);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
llook_at_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==10){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float eyeX=lua_tonumber(L,2);
		float eyeY=lua_tonumber(L,3);
		float eyeZ=lua_tonumber(L,4);
		float centerX=lua_tonumber(L,5);
		float centerY=lua_tonumber(L,6);
		float centerZ=lua_tonumber(L,7);
		float upX=lua_tonumber(L,8);
		float upY=lua_tonumber(L,9);
		float upZ=lua_tonumber(L,10);
		if(!isGLM)
			mat_look_at(m,eyeX,eyeY,eyeZ,centerX, centerY,centerZ, 
					upX,upY,upZ);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			mat_m=glm::lookAtRH(glm::vec3(eyeX,eyeY,eyeZ),
							glm::vec3(centerX,centerY,centerZ),glm::vec3(upX,upY,upZ));
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lortho_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==7){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float left=lua_tonumber(L,2);
		float right=lua_tonumber(L,3);
		float bottom=lua_tonumber(L,4);
		float top=lua_tonumber(L,5);
		float near=lua_tonumber(L,6);
		float far=lua_tonumber(L,7);
		if(isGLM)
			mat_ortho(m,left,right,bottom,top,near,far);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			mat_m=glm::orthoRH(left,right,bottom,top,near,far);
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lfrustum_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==7){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float left=lua_tonumber(L,2);
		float right=lua_tonumber(L,3);
		float bottom=lua_tonumber(L,4);
		float top=lua_tonumber(L,5);
		float near=lua_tonumber(L,6);
		float far=lua_tonumber(L,7);
		if(isGLM)
			mat_frustum(m,left,right,bottom,top,near,far);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			glm::frustumRH(left,right,bottom,top,near,far);
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lperspective_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==5){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float fovy=lua_tonumber(L,2);
		float aspect=lua_tonumber(L,3);
		float near=lua_tonumber(L,4);
		float far=lua_tonumber(L,5);
		if(isGLM)
			mat_perspective(m,fovy,aspect,near,far);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			mat_m=glm::perspectiveRH(fovy/180* 3.14159265358979323f,aspect, near,far);
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
ltranslate_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==4){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float x=lua_tonumber(L,2);
		float y=lua_tonumber(L,3);
		float z=lua_tonumber(L,4);
		if(!isGLM)
			mat_translate(m,x,y,z);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			mat_m=glm::translate(mat_m,glm::vec3(x,y,z));
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lscale_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==4){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float x=lua_tonumber(L,2);
		float y=lua_tonumber(L,3);
		float z=lua_tonumber(L,4);
		if(!isGLM)
			mat_scale(m,x,y,z);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			mat_m=glm::scale(mat_m,glm::vec3(x,y,z));
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lrotate_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==5){
		struct matrix *m=(matrix *)lua_touserdata(L,1);
		float r=lua_tonumber(L,2);
		float x=lua_tonumber(L,3);
		float y=lua_tonumber(L,4);
		float z=lua_tonumber(L,5);
		if(!isGLM)
			mat_rotate(m,r,x,y,z);
		else{
			glm::mat4x4 mat_m;
			assign_mat(mat_m,m);
			mat_m=glm::rotate(mat_m,r,glm::vec3(x,y,z));
			rassign_mat(m,mat_m);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
ladd(lua_State *L){
	int n=lua_gettop(L);
	if(n==3){
		struct vector *v=(vector *)lua_touserdata(L,1);
		struct vector *v1=(vector *)lua_touserdata(L,2);
		struct vector *v2=(vector *)lua_touserdata(L,3);
		if(isGLM)
			vec_add(v,v1,v2);
		else{
			glm::vec4 vec_v,vec_v1,vec_v2;
			assign_vec(vec_v,v);assign_vec(vec_v1,v1);assign_vec(vec_v2,v2);
			vec_v=vec_v1+vec_v2;
			rassign_vec(v,vec_v);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lsub(lua_State *L){
	int n=lua_gettop(L);
	if(n==3){
		struct vector *v=(vector *)lua_touserdata(L,1);
		struct vector *v1=(vector *)lua_touserdata(L,2);
		struct vector *v2=(vector *)lua_touserdata(L,3);
		if(isGLM)
			vec_sub(v,v1,v2);
		else{
			glm::vec4 vec_v,vec_v1,vec_v2;
			assign_vec(vec_v,v);assign_vec(vec_v1,v1);assign_vec(vec_v2,v2);
			vec_v=vec_v1-vec_v2;
			rassign_vec(v,vec_v);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
ldot(lua_State *L){
	float result;
	int n=lua_gettop(L);
	if(n==2){
		struct vector *v1=(vector *)lua_touserdata(L,1);
		struct vector *v2=(vector *)lua_touserdata(L,2);\
		if(isGLM)
			vec_dot(&result,v1,v2);
		else{
			glm::vec4 vec_v1,vec_v2;
			assign_vec(vec_v1,v1);assign_vec(vec_v2,v2);
			result=glm::dot(vec_v1,vec_v2);
		}
	}
	lua_pushnumber(L,result);
	return 1;
}

static int
lcross(lua_State *L){
	int n=lua_gettop(L);
	if(n==2){
		struct vector *vresult=(vector *)lua_touserdata(L,1);
		struct vector v1;
		struct vector *v2=(vector *)lua_touserdata(L,2);
		v1=*vresult;
		if(isGLM)
			vec_cross(vresult,&v1,v2);
		else{
			glm::vec3 vec_vresult,vec_v1,vec_v2;
			assign_vec3(vec_v1,&v1);
			assign_vec3(vec_v2,v2);
			vec_vresult=glm::cross(vec_v1,vec_v2);
			rassign_vec3(vresult,vec_vresult);
		}
	} else if(n==3){
		struct vector *vresult=(vector *)lua_touserdata(L,1);
		struct vector *v1=(vector *)lua_touserdata(L,2);
		struct vector *v2=(vector *)lua_touserdata(L,3);
		if(isGLM)
			vec_cross(vresult,v1,v2);
		else{
			glm::vec3 vec_vresult,vec_v1,vec_v2;
			assign_vec3(vec_vresult,vresult);
			assign_vec3(vec_v1,v1);
			assign_vec3(vec_v2,v2);
			vec_vresult=glm::cross(vec_v1,vec_v2);
			rassign_vec3(vresult,vec_vresult);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lnormalize(lua_State *L){
	int n=lua_gettop(L);
	if(n==2){
		struct vector *vresult=(vector *)lua_touserdata(L,1);
		struct vector *v=(vector *)lua_touserdata(L,2);
		if(isGLM)
			vec_normalize(vresult,v);
		else{
			glm::vec4 vec_vresult;
			glm::vec4 vec_v;
			assign_vec(vec_vresult,vresult);
			assign_vec(vec_v,v);
			vec_vresult=glm::normalize(vec_v);
			rassign_vec(vresult,vec_vresult);
		}
	}
	return 1;
}

static int
lunpack_vec(lua_State *L){
	int n=lua_gettop(L);
	if(n==1){
		struct vector *v=(vector *)lua_touserdata(L,1);
		lua_pushnumber(L,v->x);
		lua_pushnumber(L,v->y);
		lua_pushnumber(L,v->z);
	}
	return 3;
}

static int
lidentity_mat(lua_State *L){
	struct matrix *m=(matrix *)lua_touserdata(L,1);
	if(!isGLM)
		mat_identity(m);
	else{
		glm::mat4x4 mat_m(1);
		rassign_mat(m,mat_m);
	}
	return 1;
}

static int
lnew_vec(lua_State *L){
	struct vector *v=(vector*)lua_newuserdata(L,sizeof(struct vector));
	v->x=0.0f;v->y=0.0f;v->z=0.0f;v->w=1.0f;
	return 1;
}

static int
lnew_quat(lua_State *L){
	struct quaternion *q=(quaternion *)lua_newuserdata(L,sizeof(struct quaternion));
	q->x=0.0f;q->y=0.0f;q->z=0.0f;q->w=1.0f;
	return 1;
}

static int
lnew_mat(lua_State *L){
	struct matrix *m=(matrix *)lua_newuserdata(L,sizeof(struct matrix));
	mat_identity(m);
	return 1;
}

static int
ldescription_mat(lua_State *L){
	struct matrix *m=(matrix *)lua_touserdata(L,1);
	char mem[512];
	sprintf(mem,"matrix={\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f}\n",
			m->mat[0][0],m->mat[1][0],m->mat[2][0],m->mat[3][0],
			m->mat[0][1],m->mat[1][1],m->mat[2][1],m->mat[3][1],
			m->mat[0][2],m->mat[1][2],m->mat[2][2],m->mat[3][2],
			m->mat[0][3],m->mat[1][3],m->mat[2][3],m->mat[3][3]);
	lua_pushstring(L,mem);
	return 1;
}

static int
ldescription_vec(lua_State *L){
	struct vector *v=(vector *)lua_touserdata(L,1);
	char mem[512];
	sprintf(mem,"vector={%.4f,%.4f,%.4f,%.4f}\n",v->x,v->y,v->z,v->w);
	lua_pushstring(L,mem);
	return 1;
}

static int
ldescription_quat(lua_State *L){
	struct quaternion *q=(quaternion *)lua_touserdata(L,1);
	char mem[512];
	sprintf(mem,"quaternion={%.4f,%.4f,%.4f,%.4f}\n",q->x,q->y,q->z,q->w);
	lua_pushstring(L,mem);
	return 1;
}

static int
lset_vec(lua_State *L){
	float x,y,z,w;
	int n=lua_gettop(L);
	struct vector *v=(vector *)lua_touserdata(L,1);
	if(n==4){
		x=lua_tonumber(L,2);
		y=lua_tonumber(L,3);
		z=lua_tonumber(L,4);
		w=1.0f;
	}else if(n==5){
		x=lua_tonumber(L,2);
		y=lua_tonumber(L,3);
		z=lua_tonumber(L,4);
		w=lua_tonumber(L,5);
	}
	v->x=x;v->y=y;v->z=z;v->w=w;
	lua_settop(L,1);
	return 1;
}

static int
lset_quat(lua_State *L){
	float x,y,z,w;
	int n=lua_gettop(L);
	struct quaternion *q=(quaternion *)lua_touserdata(L,1);
	if(n==4){
		x=lua_tonumber(L,2);
		y=lua_tonumber(L,3);
		z=lua_tonumber(L,4);
	}else if(n==5){
		x=lua_tonumber(L,2);
		y=lua_tonumber(L,3);
		z=lua_tonumber(L,4);
		w=lua_tonumber(L,5);
	}
	q->x=x;q->y=y;q->z=z;q->w=w;
	lua_settop(L,1);
	return 1;
}

static int
lset_mat(lua_State *L){
	int n=lua_gettop(L);
	struct matrix *m=(matrix *)lua_touserdata(L,1);
	if(n==2){
		float v=lua_tonumber(L,2);
		mat_identity(m);
		m->mat[0][0]=v;m->mat[1][1]=v;m->mat[2][2]=v;m->mat[3][3]=v;
	}else if(n==17){
		for(int i=0;i<16;i++){
			int y=i/4,x=i%4;
			m->mat[x][y]=lua_tonumber(L,i+2);
		}
	}
	lua_settop(L,1);
	return 1;
}

static int
lcopy_quat(lua_State *L){
	int n=lua_gettop(L);
	if(n==2){
		struct quaternion *q1=(quaternion *)lua_touserdata(L,1);
		struct quaternion *q2=(quaternion *)lua_touserdata(L,2);
		*q1=*q2;
	}
	return 1;
}

static int
lcopy_vec(lua_State *L){
	int n=lua_gettop(L);
	if(n==2){
		struct vector *v1=(vector *)lua_touserdata(L,1);
		struct vector *v2=(vector *)lua_touserdata(L,2);
		*v1=*v2;
	}
	return 1;
}

static int
lcopy_mat(lua_State *L){
	int n=lua_gettop(L);
	if(n==2){
		struct matrix *m1=(matrix *)lua_touserdata(L,1);
		struct matrix *m2=(matrix *)lua_touserdata(L,2);
		*m1=*m2;
	}
	return 1;
}

static int
luse_glm(lua_State *L){
	int n=lua_gettop(L);
	if(n==1){
		int val=lua_tointeger(L,1);
		isGLM=val;
		lua_pushinteger(L,isGLM);
	}
	return 1;
}

extern "C"{
	int
	luaopen_math3d(lua_State *L){
		luaL_checkversion(L);

		luaL_Reg l[] = {
			{ "new_vec", lnew_vec },
			{ "new_quat", lnew_quat},
			{ "new_mat", lnew_mat },
			{ "add" ,ladd},
			{ "sub" ,lsub},
			{ "dot", ldot },
			{ "cross", lcross },
			{ "normalize" ,lnormalize},
			{ "mul_mat", lmul_mat},
			{ "mul_vec", lmul_vec },
			{ "translate_mat", ltranslate_mat },
			{ "scale_mat", lscale_mat },
			{ "rotate_mat", lrotate_mat },
			{ "look_at_mat" , llook_at_mat},
			{ "ortho_mat" , lortho_mat },
			{ "frustum_mat" , lfrustum_mat },
			{ "perspective_mat", lperspective_mat },
			{ "identity_mat", lidentity_mat }, 
			{ "description_mat" , ldescription_mat },
			{ "description_vec" , ldescription_vec },
			{ "description_quat", ldescription_quat },
			{ "set_vec" , lset_vec},
			{ "set_quat" , lset_quat},
			{ "set_mat" , lset_mat },
			{ "copy_vec", lcopy_vec},
			{ "copy_mat", lcopy_mat},
			{ "copy_quat", lcopy_quat},
			{ "unpack_vec", lunpack_vec},
			{ "use_glm",luse_glm },
			{ NULL, NULL },
		};

		luaL_newlib(L,l);
		
		return 1;
	}
}