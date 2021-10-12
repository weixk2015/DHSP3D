import numpy as np
def midpoint_upsampler(faces,vs,face_uv=None,uv=None):
    def midpoint(faces,vs,size = 3):
        vs = vs.tolist()
        #faces = faces.tolist()
        face_num = len(faces)
        new_faces = []
        dic = {}
        for  i in range(face_num):
            set_dic = {}
            p_list = []
            for j in range(3):
                p1 = faces[i][j]
                p2 = faces[i][(j+1)%3]
                tp = (p1,p2)
                if tp not in dic:
                    tmp_vs = [(vs[p1][k]+vs[p2][k])/2 for k in range(size)]
                    vs.append(tmp_vs)
                    dic[(p1,p2)] = len(vs) - 1
                    dic[(p2,p1)] = len(vs) - 1
                    p3 = len(vs) - 1
                else :
                    p3 = dic[tp]
                tmpset = set()
                tmpset.add(p1)
                tmpset.add(p2)
                set_dic[p3] = tmpset
                p_list.append(p3)
            new_faces.append(p_list)
            for j in range(3):
                p1 = p_list[j]
                p2 = p_list[(j+1)%3]
                p3 = -1
                for k in range(3):
                    if (faces[i][k] in set_dic[p1]) and (faces[i][k] in set_dic[p2]):
                        p3 = faces[i][k]
                new_faces.append([p2,p1,p3])
        return np.array(new_faces),np.array(vs)

    faces,vs = midpoint(faces,vs)
    if uv is not None:
        face_uv,uv = midpoint(face_uv,uv,size=2)
        return faces,vs,face_uv,uv
    else:
        return faces,vs