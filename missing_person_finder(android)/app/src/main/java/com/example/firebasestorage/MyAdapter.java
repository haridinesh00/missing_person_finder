package com.example.firebasestorage;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.ArrayList;

public class
MyAdapter extends RecyclerView.Adapter<MyAdapter.MyViewHolder> {

    private ArrayList<Model> mList;

    private Context context;

    public MyAdapter(Context context , ArrayList<Model> mList){

        this.context = context;
        this.mList = mList;
    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(context).inflate(R.layout.item , parent ,false);
        return new MyViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {
        Glide.with(context).load(mList.get(position).getImageUrl()).into(holder.imageView);
        holder.nametv.setText(mList.get(position).getFirstName());
        holder.desctv.setText(mList.get(position).getDescription());
        holder.lasttv.setText(mList.get(position).getLastName());
        holder.countv.setText(mList.get(position).getCountry());

    }

    @Override
    public int getItemCount() {
        return mList.size();
    }
    public static class MyViewHolder extends RecyclerView.ViewHolder{
        //View view;

        ImageView imageView;
        TextView nametv;
        TextView desctv;
        TextView lasttv;
        TextView countv;
        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            //view = itemView;
            imageView = itemView.findViewById(R.id.m_image);
            nametv = itemView.findViewById(R.id.firstname);
            desctv = itemView.findViewById(R.id.description);
            lasttv = itemView.findViewById(R.id.lastname);
            countv = itemView.findViewById(R.id.country);
        }

        /**
        public void setView(Context context, String name, String description){
            TextView nametv = view.findViewById(R.id.name);
            TextView desctv = view.findViewById(R.id.description);

            nametv.setText(name);
            desctv.setText(description);
        }
        */
    }
}