#include <iostream>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cmath>
#include <cstring>

using namespace std;

int ITER = 1;
float CONNECT_THRESHOLD = 0.5f;
const float CONNECT_THRESHOLD_ITER1 = 0.97f;
const float CONNECT_THRESHOLD_ITER2 = 0.7f;
const float CONNECT_THRESHOLD_ITER3 = 0.3f;
const int MINIMUM_INSTANCE_PIXEL = 256;
const float MINIMUM_INSTANCE_SCORE = 0.4f;
const int MININUM_INSTANCE_REGARDLESS_SCORE = 256;

struct Path
{
  float val;
  int source, target;
  int num;
  Path(float _val, int _source, int _target, int _num)
  {
    val = _val;
    source = _source;
    target = _target;
    num = _num;
  }
  Path()
  {
    val = 0;
    source = 0;
    target = 0;
    num = 0;
  }
  Path(const Path& path)
  {
    val = path.val;
    source = path.source;
    target = path.target;
    num = path.num;
  }
  const Path& operator = (const Path& path)
  {
    val = path.val;
    source = path.source;
    target = path.target;
    num = path.num;

    return *this;
  }
};

class Cmp
{
public:
  bool operator()(const Path &a, const Path &b)
  {
    return a.val < b.val;
  }
};

const int x_offset[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
const int y_offset[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

int *strides_buffer;
float *affinity_prob_buffer;
float *class_prob_buffer;
uint8_t *semantic_mask_buffer;
bool *background_mask;
uint8_t *instance_mask;
float *instance_confidence;

vector<int> *connections;
std::unordered_map<int, Path> *graph;
int *inst_size;
Path *inst_score;
int *father;
int *cur_path;
priority_queue<Path, vector<Path>, Cmp> heap;

int num_stride, image_height, image_width, num_neighbor;

int get_father(int x)
{
  if (father[x] == x)
    return x;
  father[x] = get_father(father[x]);
  return father[x];
}

void add_path_in_graph(float val, int source, int target, int num)
{
  if (source >= target)
  {
    swap(source, target);
  }
  graph[source][target] = Path(val, source, target, num);
}

Path& get_path_in_graph(int source, int target)
{
  if (source < target)
  {
    return graph[source][target];
  }
  else
  {
    return graph[target][source];
  }
}

bool has_path_in_graph(int source, int target)
{
  if (source >= target)
  {
    swap(source, target);
  }
  if (graph[source].find(target) != graph[source].end())
  {
    return true;
  }
  else
  {
    return false;
  }
}

float transform_class_prob(float input)
{
  return (1.0f / (1.0f + exp(-5.0f * input)) - 0.5f) * 2.0f;
}

float transform_stride_64(float input)
{
  return (1.0f / (1.0f + exp(-5.0f * input)) - 0.5f) * 2.0f;
}

void update_heap(int source)
{
  int target;
  int t_int = -1;
  float maximum_val = -1.0f;

  for (int i = 0; i < connections[source].size(); i++)
  {
    target = connections[source][i];
    if (source >= target)
    {
      continue;
    }
    if (target != get_father(target))
    {
      continue;
    }
    Path &source_target_path = get_path_in_graph(source, target);
    if (source_target_path.val > maximum_val)
    {
      maximum_val = source_target_path.val;
      t_int = target;
    }
  }
  if (t_int != -1 && maximum_val > CONNECT_THRESHOLD && cur_path[source] != t_int)
  {
    cur_path[source] = t_int;
    heap.push(get_path_in_graph(source, t_int));
  }
}

void calc_inst_scores(vector<vector<pair<int, float>>> &in_4,
  vector<vector<pair<int, float>>> &in_8,
  vector<vector<pair<int, float>>> &out_4,
  vector<vector<pair<int, float>>> &out_8,
  bool is_bike_prob_refine)
{
  int row_offset, col_offset;
  int t_int, t_x, t_y;
  int source, target;
  float t_float;

  for (int i = 0; i < image_height; i++)
  {
    row_offset = i * image_width * num_stride * num_neighbor;
    for (int j = 0; j < image_width; j++)
    {
      col_offset = j * num_stride * num_neighbor;
      source = i * image_width + j;
      for (int k = 0; k < num_stride; k++)
      {
        for (int l = 0; l < num_neighbor; l++)
        {
          t_int = instance_mask[source] - 1;
          if (t_int < 0) continue;

          t_float = affinity_prob_buffer[row_offset + col_offset + k * num_neighbor + l];

          t_x = i + strides_buffer[k] * x_offset[l];
          t_y = j + strides_buffer[k] * y_offset[l];
          if (t_x >= 0 && t_x < image_height && t_y >= 0 && t_y < image_width)
          {
            target = t_x * image_width + t_y;
            if (instance_mask[source] == instance_mask[target])
            {
              in_8[t_int][k].first++;
              in_8[t_int][k].second += t_float;
              if (l == 1 || l == 3 || l == 4 || l == 6)
              {
                in_4[t_int][k].first++;
                in_4[t_int][k].second += t_float;
              }
            }
            else
            {
              out_8[t_int][k].first++;
              out_8[t_int][k].second += t_float;
              if (l == 1 || l == 3 || l == 4 || l == 6)
              {
                out_4[t_int][k].first++;
                out_4[t_int][k].second += t_float;
              }
            }
          }
        }
      }
    }
  }
}

void parse_input_file(const char *INPUT_FILE)
{
  FILE *file_ptr = fopen(INPUT_FILE, "rb");
  size_t readLenth = fread(&image_height, sizeof(image_height), 1, file_ptr);
  readLenth = fread(&image_width, sizeof(image_width), 1, file_ptr);
  readLenth = fread(&num_stride, sizeof(num_stride), 1, file_ptr);
  readLenth = fread(&num_neighbor, sizeof(num_neighbor), 1, file_ptr);

  strides_buffer = new int[num_stride];
  int prob_buffer_size = image_height * image_width * num_stride * num_neighbor;
  affinity_prob_buffer = new float[prob_buffer_size];
  class_prob_buffer = new float[prob_buffer_size];
  semantic_mask_buffer = new uint8_t[image_height * image_width];

  readLenth = fread(strides_buffer, sizeof(int), num_stride, file_ptr);
  readLenth = fread(affinity_prob_buffer, sizeof(float), prob_buffer_size, file_ptr);
  readLenth = fread(class_prob_buffer, sizeof(float), prob_buffer_size, file_ptr);
  readLenth = fread(semantic_mask_buffer, sizeof(uint8_t), image_height * image_width, file_ptr);
  fclose(file_ptr);
}

void initialize_storage()
{
  connections = new vector<int>[image_height * image_width];
  graph = new std::unordered_map<int, Path>[image_height * image_width];

  inst_size = new int[image_height * image_width];
  inst_score = new Path[image_height * image_width];
  father = new int[image_height * image_width];
  cur_path = new int[image_height * image_width];

  int source;
  for (int i = 0; i < image_height; i++)
  {
    for (int j = 0; j < image_width; j++)
    {
      source = i * image_width + j;
      father[source] = source;
      inst_size[source] = 1;
    }
  }

  background_mask = new bool[image_height * image_width];
  instance_mask = new uint8_t[image_height * image_width];
}

void preprocess_instance_prob(bool is_bike_prob_refine)
{
  for (int i = 0; i < image_height; i++)
  {
    int row_offset = i * image_width * num_stride * num_neighbor;
    for (int j = 0; j < image_width; j++)
    {
      int col_offset = j * num_stride * num_neighbor;
      int source = i * image_width + j;
      for (int k = 0; k < num_stride; k++)
      {
        for (int l = 0; l < num_neighbor; l++)
        {
          int index = row_offset + col_offset + k * num_neighbor + l;
          float t_float = affinity_prob_buffer[index];
          t_float *= transform_class_prob(class_prob_buffer[index]);
          if (is_bike_prob_refine && strides_buffer[k] > strides_buffer[num_stride - 1] * 3 / 4 &&
              semantic_mask_buffer[source] >= 7)
          {
            t_float = transform_stride_64(t_float);
          }
          affinity_prob_buffer[index] = t_float;
        }
      }
    }
  }
}

void generate_background_mask()
{
  memset(background_mask, 0, sizeof(bool) * image_height * image_width);
  for (int i = 0; i < image_height; i++)
  {
    int row_offset = i * image_width * num_stride * num_neighbor;
    for (int j = 0; j < image_width; j++)
    {
      int col_offset = j * num_stride * num_neighbor;
      int source = i * image_width + j;
      bool is_background = true;
      for (int k = 0; k < num_stride; k++)
      {
        for (int l = 0; l < num_neighbor; l++)
        {
          float t_float = affinity_prob_buffer[row_offset + col_offset + k * num_neighbor + l];
          if (t_float > CONNECT_THRESHOLD_ITER3)
          {
            is_background = false;
            break;
          }
        }
        if (!is_background) break;
      }
      background_mask[source] = is_background;
    }
  }
}

void preprocess_father()
{
  const float threshold = CONNECT_THRESHOLD_ITER1;

  for (int i = 0; i < image_height - 1; i+=2)
  {
    for (int j = 0; j < image_width - 1; j += 2)
    {
      int t_x = i;
      int t_y = j;
      int row_offset = t_x * image_width * num_stride * num_neighbor;
      int col_offset = t_y * num_stride * num_neighbor;
      if (affinity_prob_buffer[row_offset + col_offset + 4] < threshold) continue; // right
      if (affinity_prob_buffer[row_offset + col_offset + 6] < threshold) continue; // bottom
      if (affinity_prob_buffer[row_offset + col_offset + 7] < threshold) continue; // right bottom

      t_x = i + 1;
      t_y = j + 1;
      row_offset = t_x * image_width * num_stride * num_neighbor;
      col_offset = t_y * num_stride * num_neighbor;
      if (affinity_prob_buffer[row_offset + col_offset + 1] < threshold) continue; // top
      if (affinity_prob_buffer[row_offset + col_offset + 3] < threshold) continue; // left

      t_x = i + 1;
      t_y = j;
      row_offset = t_x * image_width * num_stride * num_neighbor;
      col_offset = t_y * num_stride * num_neighbor;
      if (affinity_prob_buffer[row_offset + col_offset + 2] < threshold) continue; // top right

      int source = i * image_width + j;
      father[i * image_width + j + 1] = source;
      father[(i + 1) * image_width + j] = source;
      father[(i + 1) * image_width + j + 1] = source;
    }
  }
}

void build_graph(const vector<int> &stride_idx)
{
  // set to -1
  memset(cur_path, 0xff, image_height * image_width * sizeof(int));

  for (int i = 0; i < image_height; i++)
  {
    int row_offset = i * image_width * num_stride * num_neighbor;
    for (int j = 0; j < image_width; j++)
    {
      int col_offset = j * num_stride * num_neighbor;
      int source = i * image_width + j;
      if (background_mask[source]) continue;

      int t_int = -1;
      for (int k = 0; k < stride_idx.size(); k++)
      {
        for (int l = 0; l < num_neighbor; l++)
        {
          float t_float = affinity_prob_buffer[row_offset + col_offset +
                                               stride_idx[k] * num_neighbor + l];

          int t_x = i + strides_buffer[stride_idx[k]] * x_offset[l];
          int t_y = j + strides_buffer[stride_idx[k]] * y_offset[l];
          if (t_x >= 0 && t_x < image_height && t_y >= 0 && t_y < image_width)
          {
            int target = t_x * image_width + t_y;

            if (source > target) continue;
            if (background_mask[target]) continue;

            source = get_father(source);
            target = get_father(target);

            if (source == target)
            {
              inst_score[source].val = (inst_score[source].val * inst_score[source].num + t_float) /
                                       (inst_score[source].num + 1);
              inst_score[source].num++;
            }
            else
            {
              if (has_path_in_graph(source, target))
              {
                Path &t_path = get_path_in_graph(source, target);
                t_path.val = (t_path.val * t_path.num + t_float) / (t_path.num + 1);
                t_path.num++;
              }
              else
              {
                connections[source].push_back(target);
                connections[target].push_back(source);
                add_path_in_graph(t_float, source, target, 1);
              }
              Path &t_path = get_path_in_graph(source, target);
              if (t_int == -1 || t_path.val > get_path_in_graph(source, t_int).val)
              {
                t_int = target;
              }
            }
          }
        }
      }
      if (t_int != -1 && get_path_in_graph(source, t_int).val > CONNECT_THRESHOLD)
      {
        Path &t_path = get_path_in_graph(source, t_int);
        cur_path[t_path.source] = t_path.target;
        heap.push(t_path);
      }
    }
  }
}

void connect_nodes()
{
  int source, target, target_target;
  float val;

  int cnt = 0;
  while (!heap.empty())
  {
    Path path = heap.top();
    heap.pop();

    val = path.val;
    source = path.source;
    target = path.target;

    if (source != get_father(source) || target != get_father(target))
    {
      continue;
    }
    if (get_path_in_graph(source, target).val != val)
    {
      continue;
    }

    cnt++;

    inst_size[source] += inst_size[target];
    inst_score[source].val = (inst_score[source].val * inst_score[source].num +
                              inst_score[target].val * inst_score[target].num +
                              path.val * path.num) /
                             (inst_score[source].num + inst_score[target].num + path.num);
    inst_score[source].num += inst_score[target].num + path.num;
    father[target] = source;

    for (int i = 0; i < connections[target].size(); i++)
    {
      target_target = connections[target][i];
      if (get_father(source) == get_father(target_target) ||
          target_target != get_father(target_target))
      {
        continue;
      }
      if (!has_path_in_graph(source, target_target))
      {
        connections[source].push_back(target_target);
        connections[target_target].push_back(source);

        Path &target_tt_path = get_path_in_graph(target, target_target);
        add_path_in_graph(target_tt_path.val, source, target_target, target_tt_path.num);
      }
      else
      {
        Path &source_tt_path = get_path_in_graph(source, target_target);
        Path &target_tt_path = get_path_in_graph(target, target_target);

        source_tt_path.val = source_tt_path.num * source_tt_path.val +
                             target_tt_path.num * target_tt_path.val;
        source_tt_path.num += target_tt_path.num;
        source_tt_path.val /= source_tt_path.num;
      }

      if (cur_path[target_target] != source && cur_path[target_target] != target)
      {
        continue;
      }

      update_heap(target_target);
    }

    connections[target].clear();
    graph[target].clear();

    update_heap(source);
  }
}

void generate_instance_mask(unordered_map<int, int> &instance_mappings, const char *OUTPUT_FILE)
{
  int source;
  int instance_id = 0;
  for (int i = 0; i < image_height; i++)
  {
    for (int j = 0; j < image_width; j++)
    {
      source = i * image_width + j;
      if (get_father(source) != source &&
          instance_mappings.find(get_father(source)) == instance_mappings.end())
      {
        instance_id++;
        instance_mappings[get_father(source)] = instance_id;
      }
    }
  }

  unordered_map<int, int> t_instance_mappings;
  for (auto k = instance_mappings.begin(); k != instance_mappings.end(); k++)
  {
    if (inst_size[k->first] <= MINIMUM_INSTANCE_PIXEL)
    {
      continue;
    }
    if (inst_size[k->first] <= MININUM_INSTANCE_REGARDLESS_SCORE &&
        inst_score[k->first].val <= MINIMUM_INSTANCE_SCORE)
    {
      continue;
    }
    t_instance_mappings[k->first] = k->second;
  }
  instance_mappings = t_instance_mappings;

  instance_id = 0;
  for (auto k = instance_mappings.begin(); k != instance_mappings.end(); k++)
  {
    instance_id++;
    k->second = instance_id;
  }

  memset(instance_mask, 0, sizeof(uint8_t) * image_height * image_width);
  for (int i = 0; i < image_height; i++)
  {
    for (int j = 0; j < image_width; j++)
    {
      source = i * image_width + j;
      if (instance_mappings.find(get_father(source)) != instance_mappings.end())
      {
        instance_mask[source] = instance_mappings[get_father(source)];
      }
    }
  }

  FILE *file_ptr = fopen(OUTPUT_FILE, "wb");
  fwrite(instance_mask, sizeof(uint8_t), image_height * image_width, file_ptr);
  fclose(file_ptr);
}

void generate_confidence_file(const unordered_map<int, int> &instance_mappings,
                              const char *CONFIDENCE_FILE)
{
  instance_confidence = new float[instance_mappings.size() + 1];
  instance_confidence[0] = 0;
  for (auto k = instance_mappings.begin(); k != instance_mappings.end(); k++)
  {
    instance_confidence[k->second] = inst_score[k->first].val;
  }

  FILE *file_ptr = fopen(CONFIDENCE_FILE, "wb");
  fwrite(instance_confidence, sizeof(float), instance_mappings.size() + 1, file_ptr);
  fclose(file_ptr);
}

void generate_confidence_txt(const unordered_map<int, int> &instance_mappings,
                             bool is_bike_prob_refine, const char *CONFIDENCE_TXT)
{
  vector<int> instance_sizes;
  vector<vector<pair<int, float>>> in_4, in_8, out_4, out_8;
  vector<pair<int, float>> tmp;
  for (int i = 0; i < num_stride; i++)
  {
    tmp.push_back(make_pair(0, 0));
  }
  for (int i = 0; i < instance_mappings.size(); i++)
  {
    instance_sizes.push_back(0);
    in_4.push_back(tmp);
    in_8.push_back(tmp);
    out_4.push_back(tmp);
    out_8.push_back(tmp);
  }

  calc_inst_scores(in_4, in_8, out_4, out_8, is_bike_prob_refine);
  for (auto k = instance_mappings.begin(); k != instance_mappings.end(); k++)
  {
    instance_sizes[k->second - 1] = inst_size[k->first];
  }

  FILE *file_ptr = fopen(CONFIDENCE_TXT, "w");
  for (int i = 0; i < instance_mappings.size(); i++)
  {
    fprintf(file_ptr, "Instance: %d (%d-%.5f)\n",
            i + 1, instance_sizes[i], instance_confidence[i + 1]);
    int t_int = 0;
    float t_float = 0;
    for (int j = 0; j < num_stride; j++)
    {
      fprintf(file_ptr, "Stride-%d:\t%d/%.5f\t%d/%.5f\t%d/%.5f\t%d/%.5f\n", strides_buffer[j],
        in_4[i][j].first, in_4[i][j].second / in_4[i][j].first,
        in_8[i][j].first, in_8[i][j].second / in_8[i][j].first,
        out_4[i][j].first, out_4[i][j].second / out_4[i][j].first,
        out_8[i][j].first, out_8[i][j].second / out_8[i][j].first);
      t_int += in_8[i][j].first;
      t_float += in_8[i][j].second;
    }
    fprintf(file_ptr, "%.5f\n\n", t_float / t_int);
  }
  fclose(file_ptr);
}

void release_storage()
{
  delete[] strides_buffer;
  delete[] affinity_prob_buffer;
  delete[] class_prob_buffer;
  delete[] semantic_mask_buffer;
  delete[] background_mask;
  delete[] instance_mask;
  delete[] graph;
  delete[] inst_size;
  delete[] inst_score;
  delete[] father;
  delete[] cur_path;
  delete[] instance_confidence;
}

int main(int argc, char *argv[])
{
  const char *INPUT_FILE = argv[1];
  const char *OUTPUT_FILE = argv[2];
  const char *CONFIDENCE_FILE = argv[3];
  const char *CONFIDENCE_TXT = argv[4];
  const char *BIKE_PROB_REFINE_FLAG = argv[5];

  bool is_bike_prob_refine = false;
  if (strcmp(BIKE_PROB_REFINE_FLAG, "BIKE_PROB_REFINE_TRUE") == 0)
  {
    is_bike_prob_refine = true;
  }

  parse_input_file(INPUT_FILE);
  initialize_storage();

  preprocess_father();
  preprocess_instance_prob(is_bike_prob_refine);
  generate_background_mask();

  vector<int> stride_idx;
  int cur_idx = 0;

  //iter 1
  ITER = 1;
  CONNECT_THRESHOLD = CONNECT_THRESHOLD_ITER1;
  stride_idx.clear();
  while (cur_idx < num_stride - 4)
  {
    stride_idx.push_back(cur_idx);
    cur_idx++;
  }
  build_graph(stride_idx);
  connect_nodes();

  //iter 2
  ITER = 2;
  CONNECT_THRESHOLD = CONNECT_THRESHOLD_ITER2;
  stride_idx.clear();
  while (cur_idx < num_stride - 2)
  {
    stride_idx.push_back(cur_idx);
    cur_idx++;
  }
  build_graph(stride_idx);
  connect_nodes();

  //iter 3
  ITER = 3;
  CONNECT_THRESHOLD = CONNECT_THRESHOLD_ITER3;
  stride_idx.clear();
  while (cur_idx < num_stride)
  {
    stride_idx.push_back(cur_idx);
    cur_idx++;
  }
  build_graph(stride_idx);
  connect_nodes();

  unordered_map<int, int> instance_mappings;
  generate_instance_mask(instance_mappings, OUTPUT_FILE);

  generate_confidence_file(instance_mappings, CONFIDENCE_FILE);
  generate_confidence_txt(instance_mappings, is_bike_prob_refine, CONFIDENCE_TXT);

  release_storage();

  return 0;
}
