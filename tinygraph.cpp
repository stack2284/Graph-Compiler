#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <map>
#include <algorithm>
#include <fstream>
#include <dlfcn.h>
//hardcoded batch size 
int N = 10 ; 


enum class OpType
{
    Input,
    Add,
    Mul,
    Relu
};

std ::string op_to_string(OpType type)
{
    switch (type)
    {
    case OpType ::Input:
        return "Input";
    case OpType ::Add:
        return "Add";
    case OpType ::Mul:
        return "Mul";
    case OpType ::Relu:
        return "Relu";
    default:
        return "Unknown";
    }
}

struct Node
{
    OpType type;
    std ::string name;
    std ::vector<Node *> inputs;
    std ::vector<int> shape;
    u_int64_t id;
};
class Graph;

struct Tensor
{
    Node *node;
    Graph *graph;
    std ::string name() const { return node->name; };
};

class Graph
{
    std ::vector<std ::unique_ptr<Node>> nodes;
    std ::map<int, int> memory_offsets;
    int total_memory_needed = 0;
    bool is_compiler = false;

public:
    Tensor create_node(OpType type, const std ::string &name, std ::vector<Tensor> inputs)
    {
        auto node = std ::make_unique<Node>();
        node->type = type;
        node->name = name;
        node->id = nodes.size();
        for (auto &t : inputs)
        {
            node->inputs.push_back(t.node);
        }

        Node *raw_ptr = node.get();
        nodes.push_back(std ::move(node));
        return Tensor{raw_ptr, this};
    }

    Tensor add_input(const std ::string &name)
    {
        auto node = std ::make_unique<Node>();
        node->type = OpType ::Input;
        node->name = name;
        node->id = nodes.size();
        Node *raw = node.get();
        nodes.push_back(std ::move(node));
        return Tensor{raw, this};
    }
    void print_graph()
    {
        std ::cout << "--- COMPUTATIONAL GRAPH ---" << std ::endl;
        for (const auto &node : nodes)
        {
            std ::cout << "[" << node->id << "]" << node->name
                       << " (" << op_to_string(node->type) << ")";
            if (!node->inputs.empty())
            {
                std::cout << " <- inputs: ";
                for (auto inp : node->inputs)
                {
                    std::cout << inp->name << "[" << inp->id << "] ";
                }
            }
            std::cout << std::endl;
        }
        std ::cout << "---------------------------" << std ::endl;
    }

    void compile()
    {
        std::cout << "\n--- COMPILING (Memory Planning) ---\n";
        std ::vector<uint32_t> last_use(nodes.size(), 0);
        for (uint32_t i = 0; i < nodes.size(); i++)
        {
            Node *current = nodes[i].get();
            for (Node *inp : current->inputs)
            {
                last_use[inp->id] = std ::max(last_use[inp->id], i);
            }
        }
        std ::vector<bool> memory_map;
        memory_offsets.clear();
        uint32_t total_slots = 0;
        for (uint32_t i = 0; i < nodes.size(); ++i)
        {
            Node *current = nodes[i].get();
            int64_t allocate_slot = -1;
            for (uint32_t slot = 0; slot < memory_map.size(); slot++)
            {
                if (memory_map[slot] == false)
                {
                    allocate_slot = slot;
                    memory_map[slot] = true;
                    break;
                }
            }
            if (allocate_slot == -1)
            {
                allocate_slot = memory_map.size();
                memory_map.push_back(true);
            }
            memory_offsets[current->id] = allocate_slot * 1024;
            if (memory_map.size() > total_slots)
                total_slots = memory_map.size();
            std::cout << "Step " << i << ": Computed Node [" << current->name
                      << "] -> Assigned to Slot " << allocate_slot << "\n";
            for (Node *inp : current->inputs)
            {
                /* code */
                if (last_use[inp->id] == i)
                {
                    int slot_to_free = memory_offsets[inp->id] >> 10;
                    memory_map[slot_to_free] = false;
                    std::cout << "  -> Input [" << inp->name << "] is dead. Freeing Slot " << slot_to_free << "\n";
                }
            }
        }
        total_memory_needed = memory_map.size() * 1024;
        is_compiler = true;
        std::cout << "-----------------------------------\n";
        std::cout << "Total Memory Slots Needed: " << total_slots * 1024 << "\n";
        std::cout << ">> Compilation Complete. Arena Size: " << total_memory_needed << " floats.\n";
    }

    std ::string to_c_expr(Node *node)
    {
        if (node->type == OpType ::Input)
        {
            int offset = memory_offsets.at(node->id);
            return "memory[" + std::to_string(offset) + " + i]";
        }
        std::string left = to_c_expr(node->inputs[0]);
        if (node->type == OpType::Relu)
        {
            return "std::max(0.0f, " + left + ")";
        }
        std ::string right = to_c_expr(node->inputs[1]);
        if (node->type == OpType::Add)
        {
            return "(" + left + " + " + right + ")";
        }
        if (node->type == OpType::Mul)
        {
            return "(" + left + " * " + right + ")";
        }

        // not defined
        return "UNKNOWN";
    }
    std :: string  generate_kernel(Tensor output_tensor)
    {

        // // hardcodes to test
        // std::map<int, int> offsets;
        // offsets[0] = 0;
        // offsets[1] = 1024;
        // offsets[2] = 2048;
        // int out_offset = 4096;
        // // end of hardcoding

        if (!is_compiler)
        {
            std ::cerr << "Error : You must call compiler() function before generation" << std ::endl;
        }
        int out_offset = memory_offsets.at(output_tensor.node->id);
        std::string expr = to_c_expr(output_tensor.node);
        std :: string src = "";

        src += "void fused_kernel(float* memory, int N) {\n";
        src += "    for (int i = 0; i < N; i++) {\n";
        src += "        memory[" + std::to_string(out_offset) + " + i] = " + expr + ";\n";
        src += "    }\n";
        src += "}\n";
        return src;
    }

    int get_offset(Tensor t){
        return memory_offsets.at(t.node->id); 
    }
    int get_arena_size(){
        return total_memory_needed ;
    }

};

Tensor operator+(const Tensor &a, const Tensor &b)
{
    return a.graph->create_node(OpType ::Add, "Add", {a, b});
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
    return a.graph->create_node(OpType ::Mul, "Mul", {a, b});
}

// JIT engine 

// dynamic loading begins here 
// function pointer 
typedef void (*KernelFunc)(float*, int); 

void jit_execute(Graph & graph ,  std :: string & source_code , std :: vector <std :: pair<Tensor , float>> &inputs , Tensor &output_tensor  ){
    std :: cout << "Jit compiletion beguining here" << std :: endl ;

    std :: ofstream out("jit_kernel.cpp");
    out << "#include <algorithm>\n"; 
    out << "#include <cmath>\n" ;
    // Disable C++ Name Mangling so dlsym finds it
    out << "extern \"C\" {\n";
    out << source_code; 
    out << "}\n";
    out.close(); 
    // fpic for relocatoiable positions 
    if( system("g++ -shared -fPIC -O3 jit_kernel.cpp -o jit_kernel.so") != 0){
        std :: cerr << "Compilation Failed \n"; 
        return ;
    }
    std :: cout << ">> Compilation succesful (jit_kernel.so created).\n"; 

    void * handle = dlopen("./jit_kernel.so" , RTLD_LAZY);
    int ret ; 
    if(!handle){
        std  ::  cerr << "dlopen Failed " << dlerror() << std :: endl; 
        return ;
    }

    KernelFunc kernel = (KernelFunc) dlsym(handle , "fused_kernel"); 
    if(!kernel){
        std :: cerr << "dlsym failed" << dlerror() << "\n"; 
        return; 
    }

    int arena_size = graph.get_arena_size() ;
    std :: cout << ">> Allocating Arena (" << arena_size * sizeof(float) << "bytes\n"; 
    std::vector<float> memory(arena_size);
    // hardcoded batch size 
    // int N 

    
    for(auto & it : inputs){
        Tensor t = it.first ; 
        float val = it.second ; 
        int offset = graph.get_offset(t); 
        std :: cout << ">> Initializing input [" << t.name() << "] at offset " << offset << "with value " << val << "\n"; 
        for (int i = 0; i < N; i++)
        {
            memory[offset + i] = val ;
        }
    }



    std::cout << ">> Running Fused Kernel...\n";
    kernel(memory.data(), N);

    int out_offset = graph.get_offset(output_tensor); 
    float result = memory[out_offset]; 
    std::cout << ">> [SUCCESS] Result (Index 0): " << result << "\n";
    dlclose(handle);
    // removes extra files 
    // system("rm jit_kernel.cpp jit_kernel.so");
}




// Main function

int main()
{
    Graph g;

    // inputs
    Tensor A = g.add_input("A");
    Tensor B = g.add_input("B");

    Tensor T1 = A + B;  // creates an Add node
    Tensor T2 = T1 * A;

    Tensor T3 = g.create_node(OpType::Relu, "ReluB", {B});
    Tensor Out = T2 + T3;
    

    g.compile();
    std::string src = g.generate_kernel(Out);

    std::vector<std::pair<Tensor, float>> input_data = {
        {A, 2.0f},
        {B, 3.0f},
    };
    jit_execute(g, src, input_data, Out);
}